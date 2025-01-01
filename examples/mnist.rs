use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use rand::distributions::uniform::SampleUniform;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use rmatrix_ks::{
    matrix::{
        matrix::Matrix,
        serde::{from_file, to_file},
        utils::{apply, transpose},
        vector::{basis_vector, column_vector, is_column_vector, layer_product},
    },
    number::{
        instances::{float::Float, word8::Word8},
        traits::{floating::Floating, realfloat::RealFloat},
        utils::from_integral,
    },
};

const OUTPUT_DIM: usize = 10;

pub fn tanh_l<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    assert!(is_column_vector(v));

    apply(v, |e| e.hyperbolic_tangent())
}

pub fn diff_tanh_l<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    assert!(is_column_vector(v));

    Matrix::diagonal(
        v.row,
        v.row,
        &apply(v, |e| {
            N::one() / e.hyperbolic_cosine().power(N::one() + N::one())
        })
        .linear_iter()
        .cloned()
        .collect::<Vec<N>>(),
    )
    .unwrap()
}

pub fn softmax_l<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    is_column_vector(v);

    let mut max = 1;
    for p in 2..=v.row {
        if v[(p, 1)] > v[(max, 1)] {
            max = p;
        }
    }
    let exp_v = apply(v, |e| (e - v[(max, 1)].clone()).exponential());
    let exp_sum = exp_v
        .linear_iter()
        .par_bridge()
        .cloned()
        .reduce(|| N::zero(), |a, b| a + b);
    apply(&exp_v, |e| e / exp_sum.clone())
}

pub fn diff_softmax_l<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    assert!(is_column_vector(v));

    let softmax_val = softmax_l(v);
    let diag_softmax = Matrix::diagonal(
        v.row,
        v.row,
        &softmax_val
            .linear_iter()
            .take(v.row)
            .cloned()
            .collect::<Vec<N>>(),
    )
    .unwrap();
    let diff = layer_product(&softmax_val, &transpose(&softmax_val));
    diag_softmax - diff
}

pub fn square_loss<N>(predict_v: &Matrix<N>, should: &Matrix<N>) -> N
where
    N: RealFloat,
{
    assert!(is_column_vector(predict_v) && is_column_vector(should));
    assert_eq!(predict_v.row, should.row);

    (predict_v.clone() - should.clone())
        .linear_iter()
        .par_bridge()
        .map(|e| e.clone() * e.clone())
        .reduce(|| N::zero(), |a, b| a + b)
}

/// b0, w1, b1
pub fn grad_parameters<N>(
    shape: (u32, u32),
    img: &[u8],
    label: u8,
    parameters: (Matrix<N>, Matrix<N>, Matrix<N>),
) -> (Matrix<N>, Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    let data = column_vector(
        (shape.0 * shape.1) as usize,
        &img.iter()
            .map(|&e| from_integral::<N, Word8>(Word8::of(e)))
            .collect::<Vec<N>>(),
    )
    .unwrap();
    let b0: Matrix<N> = parameters.0;
    let w1: Matrix<N> = parameters.1;
    let b1: Matrix<N> = parameters.2;
    let l0_in = data - b0;
    let l0_out = tanh_l(&l0_in);
    let l1_in = w1.clone() * l0_out.clone() - b1;
    let l1_out = softmax_l(&l1_in);

    let two = N::one() + N::one();

    let diff = basis_vector(OUTPUT_DIM, label as usize) - l1_out;
    let common = diff_softmax_l(&l1_in) * diff;

    let grad_b0 = transpose(&w1) * common.clone() * (-two.clone());
    let grad_w1 = layer_product(&common, &transpose(&l0_out)) * (-two.clone());
    let grad_b1 = common * (-two);

    (grad_b0, grad_w1, grad_b1)
}

pub fn buffer_to_usize(buf: &[u8], is_big_endian: bool) -> usize {
    let mut p = 0;
    if is_big_endian {
        for idx in (0..buf.len()).rev() {
            p = p + buf[idx] as usize * 2usize.pow(8 * ((buf.len() - idx) as u32 - 1));
        }
    } else {
        for idx in 0..buf.len() {
            p = p + buf[idx] as usize * 2usize.pow(8 * idx as u32);
        }
    }
    p
}

pub fn load_mnist_images<P>(path: P) -> (usize, u32, u32, Vec<Vec<u8>>)
where
    P: AsRef<Path> + std::fmt::Debug,
{
    let file_handle = File::open(path).unwrap();
    let mut buffer = Vec::new();
    let _ = BufReader::new(file_handle)
        .read_to_end(&mut buffer)
        .unwrap();
    let mut reader = buffer.iter().skip(4);
    let mut number_buffer = [0u8; 4];
    for p in 0..4 {
        number_buffer[p] = reader.next().unwrap().clone();
    }
    let image_count = buffer_to_usize(&number_buffer, true);
    for p in 0..4 {
        number_buffer[p] = reader.next().unwrap().clone();
    }
    let row_shape = buffer_to_usize(&number_buffer, true) as u32;
    for p in 0..4 {
        number_buffer[p] = reader.next().unwrap().clone();
    }
    let column_shape = buffer_to_usize(&number_buffer, true) as u32;
    let mnist_data = reader.cloned().collect::<Vec<u8>>();
    let mnist_data = mnist_data
        .chunks_exact((row_shape * column_shape) as usize)
        .map(|chunck| chunck.to_vec())
        .collect::<Vec<Vec<u8>>>();
    (image_count, row_shape, column_shape, mnist_data)
}

pub fn read_mnist_image(idx: usize, data: &[Vec<u8>]) -> Vec<u8> { data[idx - 1].clone() }

pub fn load_mnist_labels<P>(path: P) -> (usize, Vec<u8>)
where
    P: AsRef<Path> + std::fmt::Debug,
{
    let file_handle = File::open(path).unwrap();
    let mut buffer = Vec::new();
    let _ = BufReader::new(file_handle)
        .read_to_end(&mut buffer)
        .unwrap();
    let mut reader = buffer.iter().skip(4);
    let mut number_buffer = [0u8; 4];
    for p in 0..4 {
        number_buffer[p] = reader.next().unwrap().clone();
    }
    let label_count = buffer_to_usize(&number_buffer, true);
    let mnist_label_data = reader.cloned().collect::<Vec<u8>>();
    (label_count, mnist_label_data)
}

pub fn read_mnist_label(data: &[u8], idx: usize) -> u8 { data[idx - 1].clone() }

pub fn mnist_image_buffer_save(filename: &str, buf: &[u8], shape: (u32, u32)) {
    image::save_buffer(
        filename,
        buf,
        shape.0,
        shape.1,
        image::ExtendedColorType::L8,
    )
    .unwrap();
}

fn validate_loss<N>(
    validate_dataset: &[Vec<u8>],
    validate_labels: &[u8],
    parameters: (Matrix<N>, Matrix<N>, Matrix<N>),
) -> N
where
    N: RealFloat,
{
    validate_dataset
        .iter()
        .enumerate()
        .par_bridge()
        .map(|(idx, img)| {
            let predict_v = predict(img, parameters.clone());
            square_loss(
                &predict_v,
                &basis_vector(OUTPUT_DIM, validate_labels[idx] as usize),
            )
        })
        .reduce(|| N::zero(), |a, b| a + b)
}

fn validate_accuracy<N>(parameters: (Matrix<N>, Matrix<N>, Matrix<N>)) -> f32
where
    N: RealFloat,
{
    0.0
}

fn train<N>(batch_count: usize, validate: f32, init_boundary: (N, N))
where
    N: RealFloat + SampleUniform,
{
    // load train data
    let (count, row_shape, column_shape, mnist_data) =
        load_mnist_images("data/mnist/train-images.idx3-ubyte");
    let (_, mnist_label_data) = load_mnist_labels("data/mnist/train-labels.idx1-ubyte");
    // only two layer
    // first layer
    // l0 = A0 (data + b0)
    let mut b0 = Matrix::<N>::defaults((row_shape * column_shape) as usize, 1);
    // second layer
    // l1 = A1 (w1 . l0 + b1)
    let mut b1 = Matrix::<N>::defaults(OUTPUT_DIM, 1);
    let mut w1 = Matrix::<N>::rand(
        OUTPUT_DIM,
        (row_shape * column_shape) as usize,
        init_boundary.0,
        init_boundary.1,
    );

    // split validate dataset
    let validate_count = (count as f32 * validate).round_ties_even() as usize;
    let train_count = count - validate_count;
    let per_bat_validate = validate_count / batch_count;
    let per_bat_train = train_count / batch_count;

    // save parameters
    to_file(&b0, "data/mnist/result/layer_b0.txt");
    to_file(&b1, "data/mnist/result/layer_b1.txt");
    to_file(&w1, "data/mnist/result/layer_w1.txt");
}

fn predict<N>(img: &[u8], parameters: (Matrix<N>, Matrix<N>, Matrix<N>)) -> Matrix<N>
where
    N: RealFloat,
{
    let data = column_vector(
        img.len(),
        &img.iter()
            .map(|&e| from_integral::<N, Word8>(Word8::of(e)))
            .collect::<Vec<N>>(),
    )
    .unwrap();
    let b0: Matrix<N> = parameters.0;
    let w1: Matrix<N> = parameters.1;
    let b1: Matrix<N> = parameters.2;
    let l0_out = tanh_l(&(data - b0));
    softmax_l(&(w1 * l0_out - b1))
}

fn main() {
    let (_, row_shape, column_shape, mnist_data) =
        load_mnist_images("data/mnist/t10k-images.idx3-ubyte");
    let (_, mnist_label_data) = load_mnist_labels("data/mnist/t10k-labels.idx1-ubyte");
    let idx = 67;

    let p = read_mnist_image(idx, &mnist_data);
    mnist_image_buffer_save("data/mnist/result/test.png", &p, (row_shape, column_shape));

    // set init value range
    let boundary = Float::of(6.0 / ((row_shape * column_shape) as f32 + 10.0)).square_root();

    println!("p[{}] is {}", idx, read_mnist_label(&mnist_label_data, idx));
}
