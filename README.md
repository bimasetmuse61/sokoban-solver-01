# sokoban-solver-01

A Sokoban solver that answers a whole set of levels in one run, in as few
pushes as possible.

## The idea

The levels in this competition are built the same way: a hallway one square
wide, with one to four rooms hanging off it, each room opening onto the
hallway through a single door and holding 3–4 boxes and as many goals. A box
pushed through a door into the hallway can never be pushed back — the hallway
is one wide, so there is nowhere to stand to push it back — which means every
box stays in its room and the rooms are independent of each other. The level's
push-optimal count is just the sum of the rooms' push optima.

So the solver:

1. **Splits the level into rooms.** Removing the hallway row (the row the
   player starts on) from the open floor leaves one connected component per
   room, each with exactly one door cell next to the hallway.
2. **Solves each room on its own** with a push-optimal A* search (below).
3. **Stitches the rooms together** with plain walking moves along the hallway.

## The room search

A state is the player's square plus the sorted box squares. A move is one
push: the player first walks (BFS) to the square behind a box, then pushes it
one square. The heuristic is the minimum-cost assignment of boxes to goals
over precomputed walking distances — it never overestimates the pushes still
needed, so the first solution A* pops is push-optimal for that room. Dead
states are pruned with:

- **dead squares** — squares from which no goal can ever be reached by
  pushing (reverse reachability from the goals);
- **frozen boxes** — a box blocked on both horizontal neighbours and both
  vertical neighbours while off its goal;
- **2×2 blocks** of walls/boxes holding a box off its goal.

A room's goal test also requires the player to be able to walk back out to
the door, so the stitched walk to the next room always exists. The search is
cut off by a fixed number of expansions, never by the clock, so two runs
produce byte-identical output.

If a level does not split into rooms as described, the same search runs once
over the whole level as a fallback. A room the search gives up on makes its
whole instance unsolved (empty answer) rather than a wrong one.

## Usage

```
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only. No network, no clock reads, no randomness.
