// gkern 的 C++ 参考实现：把教学模型里最热的两个访存判据用原生代码再写一遍。
//
// 为什么要有这份代码
// ------------------
// 1. 语义对拍：Python/numpy 侧（gkern/gpgpu/memory.py）是权威定义，这份 C++ 必须
//    逐值相同——包括 Python 向下取整的除法、尾块不足 32 lane 时按实际 lane 计。
// 2. 它对应真实 GPU 工具链里那类「把索引折成事务数 / 冲突数」的纯计算内核，
//    在 CUDA 里它们是 host 侧的辅助函数，在国产卡的 profiling 工具里同构。
// 3. 用同一套语义把 GIL 释放掉跑，可以直接量出「这份计算值不值得写 C++」。
//
// 语义细节（与 memory.py 严格一致）
// --------------------------------
// - 事务计数：段 = floor(byte_offset / 128)，返回不同段的个数。
// - bank 冲突度：bank = floor(byte_offset / 4) % 32，返回单 bank 最大命中数。
// - warp 冲突数：按 32 个 lane 一组，组内 degree>1 时累加 size*(degree-1)。
// 负数偏移在 Python 里走向下取整（-1 // 128 == -1），C++ 的 / 是向零截断，
// 所以这里显式实现 floor_div——这是本文件唯一「不照抄 C++ 语义」的地方。

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace {

// Python 语义的整除（向下取整）：divisor 恒正。
inline std::int64_t floor_div(std::int64_t value, std::int64_t divisor) {
    if (value >= 0) return value / divisor;
    return -(((-value) + divisor - 1) / divisor);
}

using Int64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

std::int64_t count_transactions(const Int64Array& offsets, std::int64_t transaction_bytes) {
    if (transaction_bytes <= 0) throw py::value_error("transaction_bytes 必须为正");
    const auto n = static_cast<std::size_t>(offsets.size());
    if (n == 0) return 0;

    const auto* data = offsets.data();
    std::vector<std::int64_t> segments(n);
    for (std::size_t i = 0; i < n; ++i) {
        segments[i] = floor_div(data[i], transaction_bytes);
    }
    std::sort(segments.begin(), segments.end());
    const auto last = std::unique(segments.begin(), segments.end());
    return static_cast<std::int64_t>(last - segments.begin());
}

std::int64_t bank_conflict_degree(const Int64Array& byte_offsets,
                                  std::int64_t bank_count,
                                  std::int64_t bank_width) {
    if (bank_count <= 0) throw py::value_error("bank_count 必须为正");
    if (bank_width <= 0) throw py::value_error("bank_width 必须为正");
    const auto n = static_cast<std::size_t>(byte_offsets.size());
    if (n == 0) return 0;

    const auto* data = byte_offsets.data();
    std::vector<std::int64_t> counts(static_cast<std::size_t>(bank_count), 0);
    std::int64_t degree = 0;
    for (std::size_t i = 0; i < n; ++i) {
        // Python 的 % 对正模数恒非负，C++ 的 % 会带符号，所以先规约到非负。
        std::int64_t bank = floor_div(data[i], bank_width) % bank_count;
        if (bank < 0) bank += bank_count;
        const auto hit = ++counts[static_cast<std::size_t>(bank)];
        if (hit > degree) degree = hit;
    }
    return degree;
}

std::int64_t warp_bank_conflicts(const Int64Array& indices,
                                 std::int64_t item_bytes,
                                 std::int64_t bank_count,
                                 std::int64_t warp_size) {
    if (item_bytes <= 0) throw py::value_error("item_bytes 必须为正");
    if (warp_size <= 0) throw py::value_error("warp_size 必须为正");

    const auto n = static_cast<std::size_t>(indices.size());
    if (n == 0) return 0;

    const auto* data = indices.data();
    std::vector<std::int64_t> counts(static_cast<std::size_t>(bank_count), 0);
    std::int64_t extra = 0;

    for (std::size_t start = 0; start < n; start += static_cast<std::size_t>(warp_size)) {
        const std::size_t stop = std::min(n, start + static_cast<std::size_t>(warp_size));
        std::fill(counts.begin(), counts.end(), 0);
        std::int64_t degree = 0;
        for (std::size_t i = start; i < stop; ++i) {
            const std::int64_t bytes = data[i] * item_bytes;
            std::int64_t bank = floor_div(bytes, item_bytes) % bank_count;
            if (bank < 0) bank += bank_count;
            const auto hit = ++counts[static_cast<std::size_t>(bank)];
            if (hit > degree) degree = hit;
        }
        if (degree > 1) {
            extra += static_cast<std::int64_t>(stop - start) * (degree - 1);
        }
    }
    return extra;
}

}  // namespace

PYBIND11_MODULE(gkern_cpp, m) {
    m.doc() = "gkern 的 C++ 访存判据实现：事务计数与共享内存 bank 冲突（与 numpy 版逐值对拍）";

    m.def("count_transactions", &count_transactions,
          py::arg("byte_offsets"), py::arg("transaction_bytes") = 128,
          py::call_guard<py::gil_scoped_release>(),
          "把一组字节地址折成事务数（不同段的个数，段 = floor(offset/128)）");

    m.def("bank_conflict_degree", &bank_conflict_degree,
          py::arg("byte_offsets"), py::arg("bank_count") = 32, py::arg("bank_width") = 4,
          py::call_guard<py::gil_scoped_release>(),
          "返回单 bank 最大命中数（1 表示无冲突）");

    m.def("warp_bank_conflicts", &warp_bank_conflicts,
          py::arg("indices"), py::arg("item_bytes") = 4,
          py::arg("bank_count") = 32, py::arg("warp_size") = 32,
          py::call_guard<py::gil_scoped_release>(),
          "按 32 lane 分组统计共享内存访问的额外轮次（冲突数）");
}
