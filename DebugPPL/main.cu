﻿#include <iostream>
#include <random>
#include <memory>
#include <glm/glm.hpp>

#include "cuda/dispatcher.h"
#include "cuda/helper.cuh"
#include "shared/structures.h"
#include "shared/brt_func.h" // this is actually a CPU function

#include <gtest/gtest.h>

#include <execution> // just for speeding up the test

#include "shared/edge_func.h"

// ------------------ configs ------------------
//constexpr auto n = 1920 * 1080; // ~2M

constexpr auto n = 640 * 480; // ~300k
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// used for quickly setting up the test
constexpr auto test_case_grid_size = 16;


[[nodiscard]] std::unique_ptr<pipe> generate_pipe()
{
	auto gpu_pip = std::make_unique<pipe>(n, min_coord, range, seed);

	std::mt19937 gen(seed); // NOLINT(cert-msc51-cpp)
	std::uniform_real_distribution dis(min_coord, min_coord + range);
	std::generate_n(std::execution::seq, gpu_pip->u_points, n, [&dis, &gen]
	{
		return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
	});

	return gpu_pip;
}

// ===============================================
//	Morton + Sort
// ===============================================

void test_morton_and_sort(const int grid_size)
{
	auto gpu_pip = generate_pipe();
	const auto cpu_points = std::vector(gpu_pip->u_points, gpu_pip->u_points + n);

	// ------- testing region ------------
	constexpr auto stream_id = 0;
	gpu::dispatch_ComputeMorton(grid_size, stream_id, *gpu_pip);
	gpu::dispatch_RadixSort(grid_size, stream_id, *gpu_pip);
	gpu::sync_stream(stream_id);
	// -----------------------------------

	// generate CPU result
	std::vector<morton_t> cpu_morton(n);
	std::transform(std::execution::par, cpu_points.begin(), cpu_points.end(), cpu_morton.begin(), [&](const auto& p)
	{
		return shared::xyz_to_morton32(p, min_coord, range);
	});
	std::sort(std::execution::par, cpu_morton.begin(), cpu_morton.end());

	const auto is_sorted = std::is_sorted(gpu_pip->u_morton, gpu_pip->u_morton + n);
	EXPECT_TRUE(is_sorted);

	const auto is_equal = std::equal(cpu_morton.begin(), cpu_morton.end(), gpu_pip->u_morton);
	EXPECT_TRUE(is_equal);
}


TEST(ComputeMorton, GridSize)
{
	for (auto i = 1; i < 16; i++)
	{
		EXPECT_NO_FATAL_FAILURE(test_morton_and_sort(i));
	}
}

// ===============================================
//	Unique
// ===============================================

void test_unique(const int grid_size)
{
	auto gpu_pip = generate_pipe();

	// generate GPU result
	constexpr auto stream_id = 0;
	gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, *gpu_pip);
	gpu::dispatch_RadixSort(test_case_grid_size, stream_id, *gpu_pip);
	SYNC_DEVICE();

	// generate CPU result (assume previous test is correct)
	const std::vector cpu_morton(gpu_pip->u_morton, gpu_pip->u_morton + n);

	// ------- testing region ------------
	gpu::dispatch_RemoveDuplicates_sync(grid_size, stream_id, *gpu_pip);
	gpu::sync_stream(stream_id);
	const auto gpu_n_unique = gpu_pip->n_unique_mortons();
	// -----------------------------------

	std::vector<morton_t> cpu_morton_alt(n);
	const auto last = std::unique_copy(cpu_morton.begin(), cpu_morton.end(), cpu_morton_alt.begin());
	const auto cpu_n_unique = std::distance(cpu_morton_alt.begin(), last);

	EXPECT_EQ(cpu_n_unique, gpu_n_unique);

	for (auto i = 0; i < gpu_n_unique; i++)
	{
		EXPECT_EQ(cpu_morton_alt[i], gpu_pip->u_morton_alt[i]);
	}
}

TEST(Unique, GridSize)
{
	for (auto i = 1; i < 16; i++)
	{
		EXPECT_NO_FATAL_FAILURE(test_unique(i));
	}
}

// ===============================================
//	Binary Radix Tree
// ===============================================

void test_binary_radix_tree(const int grid_size)
{
	auto gpu_pip = generate_pipe();

	// generate GPU result
	constexpr auto stream_id = 0;
	gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, *gpu_pip);
	gpu::dispatch_RadixSort(test_case_grid_size, stream_id, *gpu_pip);
	gpu::dispatch_RemoveDuplicates_sync(test_case_grid_size, stream_id, *gpu_pip);
	SYNC_DEVICE();

	const std::vector cpu_morton(gpu_pip->u_morton_alt, gpu_pip->u_morton_alt + n);

	const auto cpu_n_brt_nodes = gpu_pip->n_brt_nodes();

	// ------- testing region ------------
	gpu::dispatch_BuildRadixTree(grid_size, stream_id, *gpu_pip);
	gpu::sync_stream(stream_id);
	// -----------------------------------

	const radix_tree cpu_tree(cpu_n_brt_nodes);

	for (auto i = 0; i < cpu_n_brt_nodes; i++)
	{
		cpu::process_radix_tree_i(i, cpu_n_brt_nodes, cpu_morton.data(), &cpu_tree);
	}

	for (auto i = 0; i < cpu_n_brt_nodes; i++)
	{
		ASSERT_EQ(cpu_tree.u_prefix_n[i], gpu_pip->brt.u_prefix_n[i])
              << "Mismatch at index " << i;

		ASSERT_EQ(cpu_tree.u_has_leaf_left[i], gpu_pip->brt.u_has_leaf_left[i]);
		ASSERT_EQ(cpu_tree.u_has_leaf_right[i], gpu_pip->brt.u_has_leaf_right[i]);
		ASSERT_EQ(cpu_tree.u_left_child[i], gpu_pip->brt.u_left_child[i]);
		ASSERT_EQ(cpu_tree.u_parent[i], gpu_pip->brt.u_parent[i]);
	}
}

TEST(BinaryRadixTree, GridSize)
{
	for (auto i = 1; i < 16; i++)
	{
		EXPECT_NO_FATAL_FAILURE(test_binary_radix_tree(i));
	}
}

// ===============================================
//	Edge Count + Offset
// ===============================================

void test_edge_count(const int grid_size)
{
	auto gpu_pip = generate_pipe();

	// generate GPU result
	constexpr auto stream_id = 0;
	gpu::dispatch_ComputeMorton(test_case_grid_size, stream_id, *gpu_pip);
	gpu::dispatch_RadixSort(test_case_grid_size, stream_id, *gpu_pip);
	gpu::dispatch_RemoveDuplicates_sync(test_case_grid_size, stream_id, *gpu_pip);
	gpu::dispatch_BuildRadixTree(test_case_grid_size, stream_id, *gpu_pip);
	SYNC_DEVICE();


	// ------- testing region ------------
	gpu::dispatch_EdgeCount(grid_size, stream_id, *gpu_pip);
	gpu::dispatch_EdgeOffset(grid_size, stream_id, *gpu_pip);
	gpu::sync_stream(stream_id);
	// -----------------------------------

	const auto cpu_n_brt_nodes = gpu_pip->n_brt_nodes();
	std::vector<int> cpu_edge_count(cpu_n_brt_nodes);
	std::vector<int> cpu_edge_offset(cpu_n_brt_nodes);

	for (auto i = 0; i < cpu_n_brt_nodes; i++)
	{
		shared::process_edge_count_i(i, gpu_pip->brt.u_prefix_n, gpu_pip->brt.u_parent, cpu_edge_count.data());
	}
	std::exclusive_scan(cpu_edge_count.begin(), cpu_edge_count.end(), cpu_edge_offset.begin(), 0);

	for (auto i = 0; i < cpu_n_brt_nodes; i++)
	{
		ASSERT_EQ(cpu_edge_count[i], gpu_pip->u_edge_count[i]) << "Mismatch at index " << i;
		ASSERT_EQ(cpu_edge_offset[i], gpu_pip->u_edge_offset[i]) << "Mismatch at index " << i;
	}
}


TEST(TestEdgeCount, GridSize)
{
	for (auto i = 1; i < 16; i++)
	{
		EXPECT_NO_FATAL_FAILURE(test_edge_count(i));
	}
}

int main(int argc, char** argv)
{
	// some setups
	constexpr auto n_streams = 1;
	gpu::initialize_dispatcher(n_streams);

	testing::InitGoogleTest(&argc, argv);
	const auto ret = RUN_ALL_TESTS();

	gpu::release_dispatcher();
	return ret;
}
