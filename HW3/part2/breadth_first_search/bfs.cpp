#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
// #define VERBOSE 0

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
    std::vector<int> &unvisited_nodes,
    std::vector<int> &next_unvisited_nodes,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances
    )
{
    std::vector<int> tmp_frontier[omp_get_max_threads()];
    int depth = distances[frontier->vertices[0]];
    // std::vector<int> removed_nodes[omp_get_max_threads()];
    std::vector<int> local_next_unvisited_nodes[omp_get_max_threads()];

    #pragma omp parallel for
    for(int node_idx = 0; node_idx < unvisited_nodes.size(); node_idx++){
        bool flag = false;
        int node = unvisited_nodes[node_idx];
        if(distances[node] != NOT_VISITED_MARKER){
            continue;
        }
        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[node + 1];
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int incoming = g->incoming_edges[neighbor];
            if(distances[incoming] == depth){
                distances[node] = depth + 1;
                // removed_nodes[omp_get_thread_num()].push_back(node);
                tmp_frontier[omp_get_thread_num()].push_back(node);
                flag = true;
                break;
            }
        }
        if(!flag){
            local_next_unvisited_nodes[omp_get_thread_num()].push_back(node);
        }
    }

    // make sure that all threads's tmp_frontier are added to new_frontier
    for (int thread_id = 0; thread_id < omp_get_max_threads(); thread_id++){
        for (int vector_idx = 0; vector_idx < tmp_frontier[thread_id].size(); vector_idx++){
            int index = new_frontier->count++;
            new_frontier->vertices[index] = tmp_frontier[thread_id][vector_idx];
        }
        for (int vector_idx = 0; vector_idx < local_next_unvisited_nodes[thread_id].size(); vector_idx++){
            next_unvisited_nodes.push_back(local_next_unvisited_nodes[thread_id][vector_idx]);
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
    std::vector<int> unvisited_nodes(graph->num_nodes);
    #pragma omp parallel for
    for(int i = 0; i < graph->num_nodes; i++){
        unvisited_nodes[i] = i;
    }
    std::vector<int> next_unvisited_nodes;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, unvisited_nodes, next_unvisited_nodes, frontier, new_frontier, sol->distances);
        // #pragma omp parallel for
        // for(int node_idx = 0; node_idx < unvisited_nodes.size(); node_idx++){
        //     int node = unvisited_nodes[node_idx];
        //     if(sol->distances[node] != NOT_VISITED_MARKER){
        //         unvisited_nodes.erase(unvisited_nodes.begin() + node_idx);
        //         node_idx--;
        //     }
        // }

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        //swap unvisited_nodes
        unvisited_nodes.clear();
        unvisited_nodes = next_unvisited_nodes;
        next_unvisited_nodes.clear();
    }
    free(list1.vertices);
    free(list2.vertices);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
