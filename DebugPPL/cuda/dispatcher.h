#pragma once

#include "shared/structures.h"

struct pipe;

namespace gpu
{
	void initialize_dispatcher(int n_streams);
	void release_dispatcher();
	void sync_device();
	void sync_stream(int stream_id);

	void dispatch_ComputeMorton(int grid_size,
	                            int stream_id,
	                            pipe& pipe);

	void dispatch_RadixSort(int grid_size, int stream_id, pipe& pipe);

	// void dispatch_RemoveDuplicates(int grid_size,
	//                                int stream_id,
	//                                pipe& pipe);

	void dispatch_RemoveDuplicates_async(int grid_size,
	                                     int stream_id,
	                                     pipe& pipe);

	void RemoveDuplicates_on_complete(int grid_size,
	                                  int stream_id,
	                                  pipe& pipe);

	void dispatch_RemoveDuplicates_sync(int grid_size,
	                                    int stream_id,
	                                    pipe& pipe);

	void dispatch_BuildRadixTree(int grid_size,
	                             int stream_id,
	                             pipe& pipe);

	void dispatch_EdgeCount(int grid_size, int stream_id, pipe& pipe);

	void dispatch_EdgeOffset(int grid_size,
	                         int stream_id,
	                         pipe& pipe);

	void dispatch_BuildOctree(int grid_size, int stream_id, pipe& pipe);
} // namespace gpu
