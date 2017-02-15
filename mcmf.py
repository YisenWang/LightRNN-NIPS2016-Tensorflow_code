from ortools.graph import pywrapgraph

import random

import numpy as np

# Note that the value of loss must be integer

def MCMF(word_to_id, loss_r, loss_c):
  # word_to_id is the old dict, loss is a 2D array

  # Instantiate a SimpleMinCostFlow solver
  min_cost_flow = pywrapgraph.SimpleMinCostFlow()

  # define diredted graph
  # start_nodes, end_nodes, capacities, costs  

  source = 0
  sink = 2*len(word_to_id)+1

  # add arc src to vocab
  # add node supplies
  for i, word in enumerate(word_to_id.keys()):
    min_cost_flow.AddArcWithCapacityAndUnitCost(source, i+1, 1, 0)
    min_cost_flow.SetNodeSupply(i+1, 0)

  min_cost_flow.SetNodeSupply(source, len(word_to_id))
  # add arc vocab to position

  cnt_dict={}

  original_cost = 0

  for i, word in enumerate(word_to_id.keys()):
    position = word_to_id[word]
    lossr = map(sum, zip(*loss_r[tuple(position)]))
    lossc = map(sum, zip(*loss_c[tuple(position)]))
    cnt = 0
    original_cost += (lossr[position[0]] + lossc[position[1]])
    for m in range(len(lossr)):
      for n in range(len(lossc)):
        if cnt >= len(word_to_id):
          break
        lossint = int(((lossr[m]+lossc[n]) + 0.0005) * 1000)
        if i == 0:
          cnt_dict[cnt] = [m, n]

        min_cost_flow.AddArcWithCapacityAndUnitCost(i+1, len(word_to_id)+1+cnt, 1, lossint)
        cnt += 1 
  print "Original costs", original_cost
  # add arc position to dst
  for j, position in enumerate(word_to_id.keys()):
    min_cost_flow.AddArcWithCapacityAndUnitCost(len(word_to_id)+1+j, sink, 1, 0)
    min_cost_flow.SetNodeSupply(len(word_to_id)+1+j, 0)

  min_cost_flow.SetNodeSupply(sink, -len(word_to_id))


  # invoke the solver
  new_word_to_id = {}
  if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
    print "After Optimization Costs = ", min_cost_flow.OptimalCost()/1000.0
    print 

    for arc in xrange(min_cost_flow.NumArcs()):
      if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:
        if min_cost_flow.Flow(arc) > 0:
          # print 'Word %d assigned to Vector_position %d.  Cost = %d' % (
          #       min_cost_flow.Tail(arc),
          #       min_cost_flow.Head(arc),
          #       min_cost_flow.UnitCost(arc))

          word = word_to_id.keys()[min_cost_flow.Tail(arc)-1]
          position = min_cost_flow.Head(arc) - len(word_to_id) - 1

          new_word_to_id[word] = cnt_dict[position]
  else:
    print "There was an issue with the min cost flow input"

  return new_word_to_id


 