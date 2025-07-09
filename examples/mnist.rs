use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use rand::{Rng, distr::uniform::SampleUniform};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rmatrix_ks::{
    matrix::{
        Matrix,
        serde::{from_file, to_file},
        utils::{apply, transpose},
        vector::{basis_vector, column_vector, is_column_vector, layer_product},
    },
    number::{
        instances::{float::Float, word::Word, word8::Word8},
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
    img: &[u8],
    label: u8,
    parameters: (&Matrix<N>, &Matrix<N>, &Matrix<N>),
) -> (Matrix<N>, Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    let data = column_vector(
        img.len(),
        &img.iter()
            .map(|&e| from_integral::<N, Word8>(Word8::of(e)) / from_integral::<N, Word>(Word::of(256)))
            .collect::<Vec<N>>(),
    )
    .unwrap();
    let b0: Matrix<N> = parameters.0.clone();
    let w1: Matrix<N> = parameters.1.clone();
    let b1: Matrix<N> = parameters.2.clone();
    let l0_in = data - b0;
    let l0_out = tanh_l(&l0_in);
    let l1_in = w1.clone() * l0_out.clone() - b1;
    let l1_out = softmax_l(&l1_in);

    let two = N::one() + N::one();

    let diff = basis_vector(OUTPUT_DIM, label as usize + 1) - l1_out;
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
            p += buf[idx] as usize * 2usize.pow(8 * ((buf.len() - idx) as u32 - 1));
        }
    } else {
        buf.iter()
            .enumerate()
            .for_each(|(idx, e)| p += *e as usize * 2usize.pow(8 * idx as u32));
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
    number_buffer
        .iter_mut()
        .take(4)
        .for_each(|p| *p = *reader.next().unwrap());
    let image_count = buffer_to_usize(&number_buffer, true);
    number_buffer
        .iter_mut()
        .take(4)
        .for_each(|p| *p = *reader.next().unwrap());
    let row_shape = buffer_to_usize(&number_buffer, true) as u32;
    number_buffer
        .iter_mut()
        .take(4)
        .for_each(|p| *p = *reader.next().unwrap());
    let column_shape = buffer_to_usize(&number_buffer, true) as u32;
    let mnist_data = reader.cloned().collect::<Vec<u8>>();
    let mnist_data = mnist_data
        .chunks_exact((row_shape * column_shape) as usize)
        .map(|chunck| chunck.to_vec())
        .collect::<Vec<Vec<u8>>>();
    (image_count, row_shape, column_shape, mnist_data)
}

pub fn read_mnist_image(data: &[Vec<u8>], idx: usize) -> Vec<u8> { data[idx].clone() }

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
    number_buffer
        .iter_mut()
        .take(4)
        .for_each(|p| *p = *reader.next().unwrap());
    let label_count = buffer_to_usize(&number_buffer, true);
    let mnist_label_data = reader.cloned().collect::<Vec<u8>>();
    (label_count, mnist_label_data)
}

pub fn read_mnist_label(data: &[u8], idx: usize) -> u8 { data[idx] }

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

fn validate_loss<N>(validate_dataset: &[Vec<u8>], validate_labels: &[u8], parameters: (&Matrix<N>, &Matrix<N>, &Matrix<N>)) -> N
where
    N: RealFloat,
{
    validate_dataset
        .iter()
        .enumerate()
        .par_bridge()
        .map(|(idx, img)| {
            let predict_v = predict(img, parameters);
            square_loss(
                &predict_v,
                &basis_vector(OUTPUT_DIM, validate_labels[idx] as usize + 1),
            )
        })
        .reduce(|| N::zero(), |a, b| a + b)
}

fn validate_accuracy<N>(
    validate_dataset: &[Vec<u8>],
    validate_labels: &[u8],
    parameters: (&Matrix<N>, &Matrix<N>, &Matrix<N>),
) -> f32
where
    N: RealFloat,
{
    validate_dataset
        .iter()
        .enumerate()
        .par_bridge()
        .filter(|(idx, img)| {
            let predict_v = predict(img, parameters);
            let mut max = 0;
            for p in 0..OUTPUT_DIM {
                if predict_v[(max + 1, 1)] < predict_v[(p + 1, 1)] {
                    max = p;
                }
            }
            validate_labels[*idx] as usize == max
        })
        .count() as f32
        / validate_dataset.len() as f32
}

fn train<N>(batch_count: usize, validate: f32, init_learn_rate: N, init_boundary: N)
where
    N: RealFloat + SampleUniform,
{
    // load train data
    let (count, row_shape, column_shape, mnist_data) = load_mnist_images("data/mnist/train-images.idx3-ubyte");
    let (_, mnist_label_data) = load_mnist_labels("data/mnist/train-labels.idx1-ubyte");

    // init parameters, only two layer
    // first layer
    // l0 = A0 (data + b0)
    let mut b0 = Matrix::<N>::defaults((row_shape * column_shape) as usize, 1);
    // second layer
    // l1 = A1 (w1 . l0 + b1)
    let mut b1 = Matrix::<N>::defaults(OUTPUT_DIM, 1);
    let mut w1 = Matrix::<N>::rand(
        OUTPUT_DIM,
        (row_shape * column_shape) as usize,
        -init_boundary.clone(),
        init_boundary,
    );
    let mut learn_rate = init_learn_rate;
    let nine_over_ten = from_integral::<N, Word8>(Word8::of(9)) / from_integral::<N, Word8>(Word8::of(10));

    // split validate dataset
    let validate_count = (count as f32 * validate).round_ties_even() as usize;
    let validate_set = &mnist_data[..validate_count];
    let validate_label_set = &mnist_label_data[..validate_count];
    let train_set = &mnist_data[validate_count..];
    let train_label_set = &mnist_label_data[validate_count..];
    let validate_batch = (validate * batch_count as f32).floor() as usize;

    // batch train
    for p in 0..((count - validate_count) / batch_count) {
        let (mut grad_b0, mut grad_w1, mut grad_b1) = (0..batch_count)
            .par_bridge()
            .map(|idx| {
                grad_parameters(
                    &read_mnist_image(&train_set[(p * batch_count)..((p + 1) * batch_count)], idx),
                    train_label_set[(p * batch_count)..((p + 1) * batch_count)][idx],
                    (&b0, &w1, &b1),
                )
            })
            .reduce(
                || {
                    (
                        Matrix::<N>::defaults((row_shape * column_shape) as usize, 1),
                        Matrix::<N>::defaults(OUTPUT_DIM, (row_shape * column_shape) as usize),
                        Matrix::<N>::defaults(OUTPUT_DIM, 1),
                    )
                },
                |(b0, w1, b1), (b0_i, w1_i, b1_i)| (b0 + b0_i, w1 + w1_i, b1 + b1_i),
            );
        let batch_count_n = from_integral::<N, Word>(Word::of(batch_count as u32));
        grad_b0 = grad_b0 / batch_count_n.clone();
        grad_w1 = grad_w1 / batch_count_n.clone();
        grad_b1 = grad_b1 / batch_count_n;

        b0 = b0 - grad_b0 * learn_rate.clone();
        w1 = w1 - grad_w1 * learn_rate.clone();
        b1 = b1 - grad_b1 * learn_rate.clone();

        if (p + 1).is_multiple_of(10) {
            learn_rate = learn_rate * nine_over_ten.clone();
        }

        let current_accuracy = validate_accuracy(
            &validate_set[(p * validate_batch)..((p + 1) * validate_batch)],
            &validate_label_set[(p * validate_batch)..((p + 1) * validate_batch)],
            (&b0, &w1, &b1),
        );
        let curren_loss = validate_loss(
            &validate_set[(p * validate_batch)..((p + 1) * validate_batch)],
            &validate_label_set[(p * validate_batch)..((p + 1) * validate_batch)],
            (&b0, &w1, &b1),
        );

        println!(
            "batch {} / {} := acc ({}) || loss ({}) [lr: {}]",
            p,
            (count - validate_count) / batch_count,
            current_accuracy,
            curren_loss,
            learn_rate
        );
    }

    // save parameters
    to_file(&b0, "data/mnist/result/layer_b0.txt");
    to_file(&b1, "data/mnist/result/layer_b1.txt");
    to_file(&w1, "data/mnist/result/layer_w1.txt");
}

fn predict<N>(img: &[u8], parameters: (&Matrix<N>, &Matrix<N>, &Matrix<N>)) -> Matrix<N>
where
    N: RealFloat,
{
    let data = column_vector(
        img.len(),
        &img.iter()
            .map(|&e| from_integral::<N, Word8>(Word8::of(e)) / from_integral::<N, Word>(Word::of(256)))
            .collect::<Vec<N>>(),
    )
    .unwrap();
    let b0: Matrix<N> = parameters.0.clone();
    let w1: Matrix<N> = parameters.1.clone();
    let b1: Matrix<N> = parameters.2.clone();
    let l0_out = tanh_l(&(data - b0));
    softmax_l(&(w1 * l0_out - b1))
}

fn main() {
    // start train
    let boundary = Float::of(6.0 / (28.0 * 28.0 + 10.0)).square_root();
    train(160, 0.05, Float::of(1.92), boundary); // comment to skip train

    // load test data
    let (count, row_shape, column_shape, test_mnist_data) = load_mnist_images("data/mnist/t10k-images.idx3-ubyte");
    let (_, test_mnist_label_data) = load_mnist_labels("data/mnist/t10k-labels.idx1-ubyte");

    // load trained parameters
    let b0: Matrix<Float> = from_file(
        (row_shape * column_shape) as usize,
        1,
        "data/mnist/result/layer_b0.txt",
    )
    .unwrap();
    let w1: Matrix<Float> = from_file(
        OUTPUT_DIM,
        (row_shape * column_shape) as usize,
        "data/mnist/result/layer_w1.txt",
    )
    .unwrap();
    let b1: Matrix<Float> = from_file(OUTPUT_DIM, 1, "data/mnist/result/layer_b1.txt").unwrap();

    // try result
    let mut rng = rand::rng();
    let idx = (rng.random::<u32>() as usize) % count;
    let p = read_mnist_image(&test_mnist_data, idx);
    mnist_image_buffer_save("data/mnist/result/test.png", &p, (row_shape, column_shape));
    let label = read_mnist_label(&test_mnist_label_data, idx);

    let pred = predict(&p, (&b0, &w1, &b1));
    let mut max = 0;
    for p in 0..OUTPUT_DIM {
        if pred[(max + 1, 1)] < pred[(p + 1, 1)] {
            max = p;
        }
    }
    println!("load label: {label}, pred: {max}");
}
