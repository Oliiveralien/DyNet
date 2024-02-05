#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <omp.h>


#ifndef TORCH_CHECK

#define TORCH_CHECK AT_CHECK

#endif


// 高版本用 TORCH_CHECK, 低版本用AT_CHECK
#define CHECK_CPU(x) TORCH_CHECK(!x.type().is_cuda(), #x, " must be a CPUtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CPU(x);       \
  CHECK_CONTIGUOUS(x)

#ifdef _OPENMP

#define DO_PRAGMA(x) _Pragma ( #x )

#define CPU_1D_KERNEL_LOOP(i, n , thread_count)             \
  DO_PRAGMA(omp parallel for num_threads(thread_count) )        \
  for (int i = 0; i < n; i++)
  
#else 

#define CPU_1D_KERNEL_LOOP(i, n)                            \
  for (int i = 0; i < n; i++)

#endif



template <typename scalar_t>
void MaskedIm2colForward(const int n, const scalar_t *data_im,
                                    const int height, const int width,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w,
                                    const int pad_h, const int pad_w,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int mask_cnt, scalar_t *data_col,
                                    const int num_thread) {
  // mask_cnt * channels
  CPU_1D_KERNEL_LOOP(index, n, num_thread) {
    // 获取被保留的像素点的坐标
    const int m_index = index % mask_cnt;
    // 依据步长和padding等信息 映射出该像素点所对应的窗口的第一个元素的坐标
    const int h_col = mask_h_idx[m_index] * stride_h - pad_h;
    const int w_col = mask_w_idx[m_index] * stride_w - pad_w;
    // 计算通道信息
    const int c_im = index / mask_cnt;  // 第几行 <cin*k^2, mask_cnt>
    const int c_col = c_im * kernel_h * kernel_w;
    // 计算当前通道的首地址，一个通道的步长是 k^2 * mask_cnt
    // 通道的首地址+m_index 就可以得到每一列的首地址，再通过加上k次mask_cnt就可以切换到第k列
    scalar_t *data_col_ptr = data_col + c_col * mask_cnt + m_index;
    for (int i = 0; i < kernel_h; ++i) {
      int h_im = h_col + i;
      for (int j = 0; j < kernel_w; ++j) {
        int w_im = w_col + j;
        // 当存在padding时, h_im和w_im会跑到图像外面，只需要判断一下就行
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          *data_col_ptr =
              (scalar_t)data_im[(c_im * height + h_im) * width + w_im];
              // 先定位通道，再定位高，再定位宽
        } else {
          *data_col_ptr = 0.0;
        }
        // 切换到下一行的位移是mask_cnt
        data_col_ptr += mask_cnt;
      }
    }
  }
}

int MaskedIm2colForwardLaucher(const at::Tensor bottom_data, const int height,
                               const int width, const int channels,
                               const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w,
                               const int pad_h, const int pad_w,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, const int mask_cnt,
                               at::Tensor top_data,
                               const int num_thread) {
  const int output_size = mask_cnt * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedIm2colLaucherForward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data<int64_t>();
        scalar_t *top_data_ = top_data.data<scalar_t>();
        MaskedIm2colForward<scalar_t>(
                output_size, bottom_data_, height, width, kernel_h, kernel_w,
                stride_h,stride_w,pad_h, pad_w, mask_h_idx_, mask_w_idx_, mask_cnt, top_data_, num_thread);
      }));
  return 1;
}

template <typename scalar_t>
void MaskedCol2imForward(const int n, const scalar_t *data_col,
                                    const int height, const int width,
                                    const int channels,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int mask_cnt, scalar_t *data_im,
                                    const int num_thread) {
  CPU_1D_KERNEL_LOOP(index, n, num_thread) {
    const int m_index = index % mask_cnt;
    const int h_im = mask_h_idx[m_index];
    const int w_im = mask_w_idx[m_index];
    const int c_im = index / mask_cnt;
    // compute the start and end of the output
    data_im[(c_im * height + h_im) * width + w_im] = data_col[index];
  }
}

int MaskedCol2imForwardLaucher(const at::Tensor bottom_data, const int height,
                               const int width, const int channels,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, 
                               const int mask_cnt,
                               at::Tensor top_data,
                               const int num_thread) {
  const int output_size = mask_cnt * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedCol2imLaucherForward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data<int64_t>();
        scalar_t *top_data_ = top_data.data<scalar_t>();

        MaskedCol2imForward<scalar_t>(
                output_size, bottom_data_, height, width, channels, mask_h_idx_,
                mask_w_idx_, mask_cnt, top_data_, num_thread);
      }));
  return 1;
}



template <typename scalar_t>
void MaskedCol2imForward1(const int n, const scalar_t *data_col,
                                    const int height, const int width,
                                    const int channels,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int64_t *c_mask_idx,
                                    const int mask_cnt, scalar_t *data_im,
                                    const int num_thread) {
  CPU_1D_KERNEL_LOOP(index, n, num_thread) {
    const int m_index = index % mask_cnt;
    const int h_im = mask_h_idx[m_index];
    const int w_im = mask_w_idx[m_index];
    const int c_im = c_mask_idx[index / mask_cnt];
    // compute the start and end of the output
    data_im[(c_im * height + h_im) * width + w_im] = data_col[index];
  }
}

int MaskedCol2imForwardLaucher1(const at::Tensor bottom_data, const int height,
                               const int width, const int channels,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx,
                               const at::Tensor c_mask_idx,
                               const int mask_cnt,
                               at::Tensor top_data,
                               const int num_thread) {
  const int output_size = mask_cnt * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bottom_data.scalar_type(), "MaskedCol2imLaucher1Forward", ([&] {
        const scalar_t *bottom_data_ = bottom_data.data<scalar_t>();
        const int64_t *mask_h_idx_ = mask_h_idx.data<int64_t>();
        const int64_t *mask_w_idx_ = mask_w_idx.data<int64_t>();
        const int64_t *c_mask_idx_ = c_mask_idx.data<int64_t>();
        scalar_t *top_data_ = top_data.data<scalar_t>();

        MaskedCol2imForward1<scalar_t>(
                output_size, bottom_data_, height, width, channels, mask_h_idx_,
                mask_w_idx_,c_mask_idx_, mask_cnt, top_data_, num_thread);
      }));
  return 1;
}




int masked_im2col_forward_cpu(const at::Tensor im, const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, 
                               const int kernel_h, const int kernel_w, 
                               const int stride_h, const int stride_w,
                               const int pad_h, const int pad_w, 
                               at::Tensor col, const int num_thread) {
                                   
  CHECK_INPUT(im);
  CHECK_INPUT(mask_h_idx);
  CHECK_INPUT(mask_w_idx);
  CHECK_INPUT(col);
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)

  int channels = im.size(1);
  int height = im.size(2);
  int width = im.size(3);
  int mask_cnt = mask_h_idx.size(0);

  MaskedIm2colForwardLaucher(im, height, width, channels, kernel_h, kernel_w,stride_h,stride_w,
                             pad_h, pad_w, mask_h_idx, mask_w_idx, mask_cnt, col, num_thread);

  return 1;
}

int masked_col2im_forward_cpu(const at::Tensor col,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx, int height,
                               int width, int channels, at::Tensor im,
                               const int num_thread) {
  CHECK_INPUT(col);
  CHECK_INPUT(mask_h_idx);
  CHECK_INPUT(mask_w_idx);
  CHECK_INPUT(im);
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)

  int mask_cnt = mask_h_idx.size(0);

  MaskedCol2imForwardLaucher(col, height, width, channels, mask_h_idx,
                             mask_w_idx, mask_cnt, im, num_thread);

  return 1;
}


int masked_col2im_forward1_cpu(const at::Tensor col,
                               const at::Tensor mask_h_idx,
                               const at::Tensor mask_w_idx,
                               const at::Tensor c_mask_idx,
                               int height, int width, 
                               int channels, at::Tensor im,
                               const int num_thread) {
  CHECK_INPUT(col);
  CHECK_INPUT(mask_h_idx);
  CHECK_INPUT(mask_w_idx);
  CHECK_INPUT(c_mask_idx);
  CHECK_INPUT(im);
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)

  int mask_cnt = mask_h_idx.size(0);

  MaskedCol2imForwardLaucher1(col, height, width, channels, mask_h_idx,
                              mask_w_idx,c_mask_idx, mask_cnt, im, num_thread);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_im2col_forward", &masked_im2col_forward_cpu,
        "masked_im2col forward (CPU)");
  m.def("masked_col2im_forward", &masked_col2im_forward_cpu,
        "masked_col2im forward (CPU)");
  m.def("masked_col2im_forward1", &masked_col2im_forward1_cpu,
        "masked_col2im forward1 (CPU)");
}

// unable to execute ':/usr/local/cuda:/usr/local/cuda/bin/nvcc': No such file or directory
// unable to execute ':/usr/local/cuda/bin/nvcc': No such file or directory