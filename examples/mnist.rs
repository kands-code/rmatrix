use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use rayon::iter::{ParallelBridge, ParallelIterator};
use rmatrix_ks::{
    matrix::{matrix::Matrix, utils::apply, vector::column_vector},
    number::{
        instances::float::Float,
        traits::{floating::Floating, realfloat::RealFloat, zero::Zero},
    },
};

pub fn tanh_l<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    if v.column == 1 {
        apply(v, |e| e.hyperbolic_tangent())
    } else {
        panic!("only vector!");
    }
}

pub fn softmax_l<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    if v.column == 1 {
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
    } else {
        panic!("noly vector!");
    }
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

pub fn load_mnist_images<P>(path: P) -> (usize, u32, u32, Vec<u8>)
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
    (image_count, row_shape, column_shape, mnist_data)
}

pub fn read_mnist_image(shape: (u32, u32), data: &[u8], idx: usize) -> Vec<u8> {
    Vec::from_iter(
        data[((idx - 1) * (shape.0 * shape.1) as usize)..(idx * (shape.0 * shape.1) as usize)]
            .iter()
            .cloned(),
    )
}

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

fn main() {
    // load train data
    let (image_count, row_shape, column_shape, mnist_data) =
        load_mnist_images("data/mnist/train-images.idx3-ubyte");
    let (labels_count, mnist_label_data) =
        load_mnist_labels("data/mnist/train-labels.idx1-ubyte");
    assert_eq!(image_count, labels_count);
    // set init value range
    let boundary = Float::of(6.0 / ((row_shape * column_shape) as f32 + 10.0)).square_root();
    // only two layer
    // first layer
    // l0 = A0 (data + b0)
    let mut b0 = Matrix::<Float>::defaults((row_shape * column_shape) as usize, 1);
    // second layer
    // l1 = A1 (w1 . l0 + b1)
    let mut b1 = Matrix::<Float>::defaults((row_shape * column_shape) as usize, 1);
    let mut w1 = Matrix::<Float>::rand(
        (row_shape * column_shape) as usize,
        10,
        -boundary.clone(),
        boundary,
    );
    println!("{}", w1);

    let idx = 98;

    let p = read_mnist_image((row_shape, column_shape), &mnist_data, idx);
    mnist_image_buffer_save("test.png", &p, (row_shape, column_shape));

    println!("p[{}] is {}", idx, read_mnist_label(&mnist_label_data, idx));

    let v = column_vector(4, &[1.0, 2.0, 3.0, 4.0].map(Float::of)).unwrap();
    let soft_v = softmax_l(&v);
    println!("{}", soft_v);
}
