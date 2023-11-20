#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <bitset>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
// #define VERBOSE 0

struct vertex_bitmap{
    int count;
    int size;
    bool *bitmap;
};

void vertex_bitmap_clear(vertex_bitmap *list)
{
    list->count = 0;
    for(int i = 0; i < list->size; i++){
        list->bitmap[i] = false;
        // list->bitmap.push_back(false);
    }
}

void vertex_bitmap_init(vertex_bitmap *list, int count)
{
    list->size = count;
    list->bitmap = (bool *)malloc(sizeof(bool) * count);
    // list->bitmap.resize(list->count);
    vertex_bitmap_clear(list);
}

inline void vertex_bitmap_set(vertex_bitmap *list, int idx)
{
    list->count++;
    list->bitmap[idx] = true;
}

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances
    )
{
    std::vector<int> tmp_frontier[omp_get_max_threads()];

    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                        ? g->num_edges
                        : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            {
                if (distances[outgoing] == NOT_VISITED_MARKER)
                // // if(__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    distances[outgoing] = distances[node] + 1;
                //     // #pragma omp critical
                //     // {
                //     //     int index = new_frontier->count++;
                //     //     new_frontier->vertices[index] = outgoing;
                //     // }
                    tmp_frontier[omp_get_thread_num()].push_back(outgoing);
                }
            }
        }
    }

    // make sure that all threads's tmp_frontier are added to new_frontier
    for (int thread_id = 0; thread_id < omp_get_max_threads(); thread_id++){
        for (int vector_idx = 0; vector_idx < tmp_frontier[thread_id].size(); vector_idx++){
            int index = new_frontier->count++;
            new_frontier->vertices[index] = tmp_frontier[thread_id][vector_idx];
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(list1.vertices);
    free(list2.vertices);
}

// Take one step of "bottom-up" BFS.  For each vertex that is NOT part
// of the frontier, but has a neighbor that is, add that vertex to the
// new_frontier.
void bottom_up_step(
    Graph g,
    vertex_bitmap *frontier,
    vertex_bitmap *new_frontier,
    int *distances
    )
{
    std::vector<int> tmp_frontier[omp_get_max_threads()];
    // int depth = distances[frontier->vertices[0]];
    int *incoming_edges = g->incoming_edges;

    #pragma omp parallel for
    for(int node_idx = 0; node_idx < g->num_nodes; node_idx++){
        int node = node_idx;
        if(distances[node] != NOT_VISITED_MARKER){
            continue;
        }
        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[node + 1];
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int incoming = incoming_edges[neighbor];
            // if(distances[incoming] == depth){
            if(frontier->bitmap[incoming]){
                distances[node] = distances[incoming] + 1;
                tmp_frontier[omp_get_thread_num()].push_back(node);
                break;
            }
        }
    }

    // make sure that all threads's tmp_frontier are added to new_frontier
    for (int thread_id = 0; thread_id < omp_get_max_threads(); thread_id++){
        for (int vector_idx = 0; vector_idx < tmp_frontier[thread_id].size(); vector_idx++){
            vertex_bitmap_set(new_frontier, tmp_frontier[thread_id][vector_idx]);
            // new_frontier->count++;
            // new_frontier->bitmap[tmp_frontier[thread_id][vector_idx]] = true;
            // new_frontier->vertices[index] = tmp_frontier[thread_id][vector_idx];
            // unvisited_nodes[tmp_frontier[thread_id][vector_idx]] = true;
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    // vertex_set list1;
    // vertex_set list2;
    // vertex_set_init(&list1, graph->num_nodes);
    // vertex_set_init(&list2, graph->num_nodes);

    vertex_bitmap list1;
    vertex_bitmap list2;
    vertex_bitmap_init(&list1, graph->num_nodes);
    vertex_bitmap_init(&list2, graph->num_nodes);

    // vertex_set *frontier = &list1;
    // vertex_set *new_frontier = &list2;
    vertex_bitmap *frontier = &list1;
    vertex_bitmap *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    // frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    frontier->bitmap[ROOT_NODE_ID] = true;
    frontier->count++;

    sol->distances[ROOT_NODE_ID] = 0;
    // std::vector<bool> unvisited_nodes(graph->num_nodes);
    // bool *unvisited_nodes = (bool *)malloc(sizeof(bool) * graph->num_nodes);
    // #pragma omp parallel for
    // for(int i = 0; i < graph->num_nodes; i++){
    //     unvisited_nodes[i] = false;
    // }

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        // vertex_set_clear(new_frontier);
        vertex_bitmap_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        // vertex_set *tmp = frontier;
        vertex_bitmap *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    // free(list1.vertices);
    // free(list2.vertices);
    free(list1.bitmap);
    free(list2.bitmap);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
