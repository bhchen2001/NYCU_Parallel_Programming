#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

  bool converged = false;
  double *score_new = (double *)malloc(sizeof(double) * numNodes);
  double tmp = 0.0, global_diff = 0.0, no_outgoing = 0.0;
  #pragma omp parallel for reduction(+:no_outgoing)
  for(int i = 0; i < numNodes; i++){
      if(outgoing_size(g, i) == 0){
          no_outgoing += damping * solution[i] / numNodes;
      }
  }
  while(!converged){
    #pragma omp parallel for
    for(int i = 0; i < numNodes; i++){
        tmp = 0.0;
        for(const Vertex *j = incoming_begin(g, i); j != incoming_end(g, i); j++){
            tmp += solution[*j]/(double)outgoing_size(g, *j);
        }
        tmp = (damping * tmp) + (1.0 - damping) / numNodes;
        tmp += no_outgoing;
        score_new[i] = tmp;
    }
    no_outgoing = 0.0;
    global_diff = 0.0;
    #pragma omp parallel for reduction(+:no_outgoing), reduction(+:global_diff)
    for(int i = 0; i < numNodes; i++){
        global_diff += fabs(score_new[i] - solution[i]);
        if(outgoing_size(g, i) == 0){
            no_outgoing += damping * score_new[i] / numNodes;
        }
    }
    memcpy(solution, score_new, sizeof(double) * numNodes); 
    converged = (global_diff < convergence);
  }
  free(score_new);
}
