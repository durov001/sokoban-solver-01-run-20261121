# Room-by-Room Push-Optimal Sokoban Solver

[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/durov001/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/durov001/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A make.fun bounty entry: a Sokoban solver for `sokoban-solver-01-run-20261121`
that answers all five instances of a set in one run, in as few pushes as it
can — a level is worth its full 1,000 points only at its push-optimal count
P*, and even one push over earns under half.

The levels are a hallway with one to four rooms off it, each room holding 3
or 4 boxes, and a box pushed out of a room's door into the hallway can never
be pushed back — so every box stays in its room and a level's P* is the sum
of its rooms' optima. The solver exploits that structure: it cuts the level
into rooms (dead squares of the hallway row cut out of the floor), solves
each room push-optimally with A* over pushes — states are the boxes plus the
player's reachable region, the heuristic is the cheapest assignment of boxes
to goals by shortest-path distance, dead squares and frozen 2×2 blocks prune
states no push can save — with a breadth-first search over pushes as
fallback, then stitches the rooms together with the player walking on from
room to room. Every budget is counted in states expanded, never in time, so
two runs answer byte for byte alike.

## Run it

```sh
node main.ts
```

It reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt`
(`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` override the directories, as in
the kit). Node 22, standard library only, no network, no clock reads.

Built for the make.fun bounty — [sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121). Referral code: zsk0hsg3.
