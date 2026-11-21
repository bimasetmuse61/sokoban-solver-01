// Sokoban solver for make.fun's sokoban-solver-01.
//
// Idea (from the level generator's structure, which the kit's RULES.txt describes):
// every level is a one-square-wide hallway with one to four rooms hanging off it,
// each room opening onto the hallway through a single door cell, holding 3-4 boxes
// and as many goals. A box pushed through a door into the hallway can never be
// pushed back (the hallway is one wide, so there is no square to stand on to push
// it back), so every box stays in its room and the rooms are independent: the
// level's push-optimal count is the sum of the rooms' push optima. This solver
// therefore
//   1. splits the level into rooms (connected components of the open floor once
//      the hallway row is removed; each room has exactly one door cell),
//   2. solves each room on its own with a push-optimal A* search, and
//   3. stitches the room solutions together with plain walking moves along the
//      hallway.
//
// The room search is a macro-move (push) A*: a state is the player's square plus
// the sorted box squares; a move is one push of one box, with the player first
// walking (BFS) to the square behind the box. The heuristic is the minimum-cost
// assignment of boxes to goals over precomputed walking distances, which never
// overestimates the pushes still needed, so the first solution A* pops is
// push-optimal for that room. Dead states are pruned with dead squares (squares
// from which no goal can ever be reached by pushing), frozen boxes (a box that
// cannot move on either axis) and 2x2 blocks of walls/boxes holding a box off a
// goal. The search is cut off by a fixed number of expansions, never by the
// clock, so two runs of the program produce byte-identical output.
//
// If a level does not split into rooms the way the generator promises, the same
// search runs once over the whole level as a fallback. A room the search gives
// up on makes its whole instance unsolved (empty answer) rather than a wrong
// one: a wrong replay scores 0 either way.
//
// Node 22 + standard library only; no network, no clock reads, no randomness.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed work budgets: determinism requires budgeting by expansions, not time.
const ROOM_EXPANSIONS = 600_000;
const TOTAL_EXPANSIONS = 4_000_000;
let spentExpansions = 0;

const DIRS: { dx: number; dy: number; ch: string }[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

/** BFS walking distances from `from` over cells where `pass` is true. -1 = unreachable. */
function bfsDist(
  from: number,
  w: number,
  h: number,
  pass: (cell: number) => boolean,
): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  if (!pass(from)) return dist;
  const q: number[] = [from];
  dist[from] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const x = c % w;
    const y = (c / w) | 0;
    if (x > 0 && pass(c - 1) && dist[c - 1] < 0) {
      dist[c - 1] = dist[c]! + 1;
      q.push(c - 1);
    }
    if (x + 1 < w && pass(c + 1) && dist[c + 1] < 0) {
      dist[c + 1] = dist[c]! + 1;
      q.push(c + 1);
    }
    if (y > 0 && pass(c - w) && dist[c - w] < 0) {
      dist[c - w] = dist[c]! + 1;
      q.push(c - w);
    }
    if (y + 1 < h && pass(c + w) && dist[c + w] < 0) {
      dist[c + w] = dist[c]! + 1;
      q.push(c + w);
    }
  }
  return dist;
}

/** Shortest walking path (as move letters) from `from` to `to`, or null. */
function bfsPath(
  from: number,
  to: number,
  w: number,
  h: number,
  pass: (cell: number) => boolean,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevCh: string[] = new Array(w * h).fill('');
  prev[from] = -1;
  const q: number[] = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    for (const d of DIRS) {
      const x = c % w;
      const y = (c / w) | 0;
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (prev[n] !== -2 || !pass(n)) continue;
      prev[n] = c;
      prevCh[n] = d.ch;
      if (n === to) {
        let s = '';
        for (let t = n; t !== from; t = prev[t]!) s = prevCh[t] + s;
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

/** Push-optimal A* over one self-contained area.
 *  `cells` are the squares boxes may occupy; `entry` is the square the player
 *  starts on (and must be able to return to, when `needExit`); boxes and goals
 *  all lie in `cells`. Returns the list of pushes [boxCell, dirIndex], or null
 *  when the expansion budget runs out. */
function solveArea(
  level: Level,
  cells: number[],
  entry: number,
  boxes0: number[],
  goals: number[],
  needExit: boolean,
  budget: number,
): { box: number; dir: number }[] | null {
  const { w, h, wall, goalSet } = level;
  const inCells = new Uint8Array(w * h);
  for (const c of cells) inCells[c] = 1;
  const pass = (c: number): boolean => (inCells[c] === 1 || c === entry) && wall[c] === 0;

  // Walking distances between every pair of area squares (ignoring boxes).
  const distFrom = new Map<number, Int32Array>();
  for (const c of cells) distFrom.set(c, bfsDist(c, w, h, pass));
  distFrom.set(entry, bfsDist(entry, w, h, pass));

  // Dead squares: a square is live when a box there can still be pushed onto
  // some goal. Reverse reachability from the goals: a box on c can be pushed
  // to neighbour n when the square beyond c (2c - n) is open for the player.
  const live = new Uint8Array(w * h);
  const liveQ: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    liveQ.push(g);
  }
  for (let qi = 0; qi < liveQ.length; qi++) {
    const n = liveQ[qi]!;
    const nx = n % w;
    const ny = (n / w) | 0;
    for (const d of DIRS) {
      const cx = nx - d.dx;
      const cy = ny - d.dy;
      if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
      const c = cy * w + cx;
      if (live[c] || !pass(c)) continue;
      const bx = cx - d.dx;
      const by = cy - d.dy;
      if (bx < 0 || bx >= w || by < 0 || by >= h) continue;
      if (!pass(by * w + bx)) continue;
      live[c] = 1;
      liveQ.push(c);
    }
  }

  // Minimum assignment of boxes to goals (brute force over permutations;
  // at most 4 boxes), cached per box configuration. Infinity when a box can
  // reach no goal at all.
  const hCache = new Map<string, number>();
  const heuristic = (boxes: number[]): number => {
    const k = boxes.join(',');
    const hit = hCache.get(k);
    if (hit !== undefined) return hit;
    let best = Infinity;
    const n = boxes.length;
    const perm = goals.slice(0, n);
    // Heap's algorithm over the first n goals (n == goals.length here).
    const gen = (m: number): void => {
      if (m === 1) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const d = distFrom.get(boxes[i]!)!.at(perm[i]!)!;
          if (d < 0) return;
          s += d;
        }
        if (s < best) best = s;
        return;
      }
      for (let i = 0; i < m; i++) {
        gen(m - 1);
        const j = m % 2 === 1 ? 0 : i;
        [perm[j], perm[m - 1]] = [perm[m - 1]!, perm[j]!];
      }
    };
    gen(n);
    hCache.set(k, best);
    return best;
  };

  const keyOf = (p: number, boxes: number[]): string => p + ':' + boxes.join(',');

  // A* state storage.
  const sp: number[] = [entry];
  const sb: number[][] = [boxes0.slice().sort((a, b) => a - b)];
  const sg: number[] = [0];
  const sparent: number[] = [-1];
  const spushBox: number[] = [-1];
  const spushDir: number[] = [-1];
  const best = new Map<string, number>([[keyOf(entry, sb[0]!), 0]]);
  // Heap entries [f, g, seq, state]; ties: deeper g first, then earlier seq.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const h0 = heuristic(sb[0]!);
  if (h0 === Infinity) return null;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hpop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  hpush([h0, 0, seq++, 0]);

  const boxAt = new Uint8Array(w * h);
  let expanded = 0;
  while (heap.length > 0 && expanded < budget && spentExpansions < TOTAL_EXPANSIONS) {
    const [, g, , s] = hpop();
    if (best.get(keyOf(sp[s]!, sb[s]!)) !== s) continue;
    const boxes = sb[s]!;
    // Goal: every box on a goal (and, when required, the player able to walk
    // back out to the entry square).
    if (boxes.every((b) => goalSet[b] === 1)) {
      if (!needExit) {
        const out: { box: number; dir: number }[] = [];
        for (let t = s; t > 0; t = sparent[t]!) out.push({ box: spushBox[t]!, dir: spushDir[t]! });
        out.reverse();
        return out;
      }
      boxAt.fill(0);
      for (const b of boxes) boxAt[b] = 1;
      const d = bfsDist(sp[s]!, w, h, (c) => pass(c) && boxAt[c] === 0);
      if (d[entry]! >= 0) {
        const out: { box: number; dir: number }[] = [];
        for (let t = s; t > 0; t = sparent[t]!) out.push({ box: spushBox[t]!, dir: spushDir[t]! });
        out.reverse();
        return out;
      }
    }
    expanded++;
    spentExpansions++;
    const p = sp[s]!;
    boxAt.fill(0);
    for (const b of boxes) boxAt[b] = 1;
    // Squares the player can reach without moving a box.
    const reach = bfsDist(p, w, h, (c) => pass(c) && boxAt[c] === 0);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di]!;
        const px = bx - d.dx;
        const py = by - d.dy;
        const dx = bx + d.dx;
        const dy = by + d.dy;
        if (px < 0 || px >= w || py < 0 || py >= h || dx < 0 || dx >= w || dy < 0 || dy >= h) continue;
        const pushCell = py * w + px;
        const dest = dy * w + dx;
        if (reach[pushCell]! < 0) continue; // player cannot get behind the box
        if (inCells[dest] !== 1 || wall[dest] === 1 || boxAt[dest] === 1) continue; // box must stay in its room
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        // Deadlock pruning.
        let dead = false;
        for (const nbCell of nb) {
          if (live[nbCell] !== 1) {
            dead = true;
            break;
          }
        }
        if (dead) continue;
        // Frozen box: blocked on both horizontal neighbours and both vertical
        // neighbours, while off its goal.
        const isBox = (c: number): boolean => nb.includes(c);
        const blk = (c: number): boolean => c < 0 || c >= w * h || wall[c] === 1 || isBox(c);
        for (const nbCell of nb) {
          if (goalSet[nbCell] === 1) continue;
          const x = nbCell % w;
          const y = (nbCell / w) | 0;
          const horiz = (x === 0 || blk(nbCell - 1)) && (x === w - 1 || blk(nbCell + 1));
          const vert = (y === 0 || blk(nbCell - w)) && (y === h - 1 || blk(nbCell + w));
          if (horiz && vert) {
            dead = true;
            break;
          }
          // 2x2 block of walls/boxes holding a box off its goal.
          for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as const) {
            const cx = x + ox;
            const cy = y + oy;
            if (cx < 0 || cy < 0 || cx + 1 >= w || cy + 1 >= h) continue;
            const c0 = cy * w + cx;
            const cells4 = [c0, c0 + 1, c0 + w, c0 + w + 1];
            if (cells4.every((c) => wall[c] === 1 || isBox(c)) && cells4.some((c) => isBox(c) && goalSet[c] !== 1)) {
              dead = true;
              break;
            }
          }
          if (dead) break;
        }
        if (dead) continue;
        const hh = heuristic(nb);
        if (hh === Infinity) continue;
        const k = keyOf(b, nb);
        const seen = best.get(k);
        if (seen !== undefined && sg[seen]! <= g + 1) continue;
        const t = sp.length;
        sp.push(b);
        sb.push(nb);
        sg.push(g + 1);
        sparent.push(s);
        spushBox.push(b);
        spushDir.push(di);
        best.set(k, t);
        hpush([g + 1 + hh, g + 1, seq++, t]);
      }
    }
  }
  return null;
}

/** Turn a push list into walk+push move letters, simulating as we go. */
function pushesToMoves(
  level: Level,
  cells: number[],
  entry: number,
  boxes0: number[],
  pushes: { box: number; dir: number }[],
): string | null {
  const { w, h, wall } = level;
  const inCells = new Uint8Array(w * h);
  for (const c of cells) inCells[c] = 1;
  const pass = (c: number): boolean => (inCells[c] === 1 || c === entry) && wall[c] === 0;
  const boxes = boxes0.slice();
  let player = entry;
  let moves = '';
  for (const { box, dir } of pushes) {
    const d = DIRS[dir]!;
    const pushCell = box - d.dy * w - d.dx;
    const boxAt = new Uint8Array(w * h);
    for (const b of boxes) boxAt[b] = 1;
    const path = bfsPath(player, pushCell, w, h, (c) => pass(c) && boxAt[c] === 0);
    if (path === null) return null;
    moves += path;
    moves += d.ch.toUpperCase();
    const at = boxes.indexOf(box);
    if (at < 0) return null;
    boxes[at] = box + d.dy * w + d.dx;
    player = box;
  }
  return moves;
}

function solve(text: string): string {
  const level = parseLevel(text);
  const { w, h, wall, boxes, player } = level;
  const hallwayRow = (player / w) | 0;

  // Split the open floor into rooms by removing the hallway row.
  const seen = new Uint8Array(w * h);
  interface Room {
    cells: number[];
    entry: number;
    boxes: number[];
    goals: number[];
  }
  const rooms: Room[] = [];
  let ok = true;
  const inRoom = new Uint8Array(w * h);
  for (let y = 0; y < h && ok; y++) {
    for (let x = 0; x < w && ok; x++) {
      const c = y * w + x;
      if (wall[c] === 1 || y === hallwayRow || seen[c] === 1) continue;
      // Flood fill one component.
      const comp: number[] = [];
      const q: number[] = [c];
      seen[c] = 1;
      for (let qi = 0; qi < q.length; qi++) {
        const cur = q[qi]!;
        comp.push(cur);
        const cx = cur % w;
        const cy = (cur / w) | 0;
        for (const d of DIRS) {
          const nx = cx + d.dx;
          const ny = cy + d.dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const n = ny * w + nx;
          if (wall[n] === 1 || ny === hallwayRow || seen[n] === 1) continue;
          seen[n] = 1;
          q.push(n);
        }
      }
      const rBoxes = comp.filter((cc) => boxes.includes(cc));
      if (rBoxes.length === 0) continue;
      const rGoals = comp.filter((cc) => level.goalSet[cc] === 1);
      if (rGoals.length !== rBoxes.length) {
        ok = false;
        break;
      }
      // The door: the component's one square next to the hallway row.
      let door = -1;
      let entry = -1;
      for (const cc of comp) {
        const cy = (cc / w) | 0;
        if (Math.abs(cy - hallwayRow) !== 1) continue;
        const n = (cy < hallwayRow ? cc + w : cc - w);
        if (wall[n] === 0 && (n / w | 0) === hallwayRow) {
          if (door !== -1) {
            ok = false;
            break;
          }
          door = cc;
          entry = n;
        }
      }
      if (door === -1) {
        ok = false;
        break;
      }
      for (const cc of comp) inRoom[cc] = 1;
      rooms.push({ cells: comp, entry, boxes: rBoxes, goals: rGoals });
    }
  }
  // Every box must belong to exactly one room, and no box may sit in the hallway.
  if (ok) {
    const covered = rooms.reduce((a, r) => a + r.boxes.length, 0);
    if (covered !== boxes.length) ok = false;
  }

  let moves = '';
  if (ok && rooms.length > 0) {
    rooms.sort((a, b) => a.entry - b.entry);
    let cur = player;
    const boxNow = boxes.slice();
    for (const room of rooms) {
      // Walk the player to this room's entry square.
      const boxAt = new Uint8Array(w * h);
      for (const b of boxNow) boxAt[b] = 1;
      const walk = bfsPath(cur, room.entry, w, h, (c) => wall[c] === 0 && boxAt[c] === 0);
      if (walk === null) return '';
      moves += walk;
      cur = room.entry;
      const pushes = solveArea(level, room.cells, room.entry, room.boxes, room.goals, true, ROOM_EXPANSIONS);
      if (pushes === null) return '';
      const roomMoves = pushesToMoves(level, room.cells, room.entry, room.boxes, pushes);
      if (roomMoves === null) return '';
      moves += roomMoves;
      // Simulate the room's pushes onto the shared box list and player.
      for (const { box, dir } of pushes) {
        const d = DIRS[dir]!;
        const at = boxNow.indexOf(box);
        if (at < 0) return '';
        boxNow[at] = box + d.dy * w + d.dx;
        cur = box;
      }
    }
  } else {
    // Fallback: one search over the whole level.
    const allCells: number[] = [];
    for (let c = 0; c < w * h; c++) if (wall[c] === 0) allCells.push(c);
    const pushes = solveArea(level, allCells, player, boxes, level.goals, false, ROOM_EXPANSIONS);
    if (pushes === null) return '';
    const m = pushesToMoves(level, allCells, player, boxes, pushes);
    if (m === null) return '';
    moves = m;
  }

  // Self-check: replay the moves; a wrong answer scores 0 either way, so
  // only hand over a verified solve.
  const simBoxes = new Set(boxes);
  let simPlayer = player;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const d = DIRS.find((dd) => dd.ch === lower);
    if (!d) return '';
    const next = simPlayer + d.dy * w + d.dx;
    if (wall[next] === 1) return '';
    if (simBoxes.has(next)) {
      const beyond = next + d.dy * w + d.dx;
      if (wall[beyond] === 1 || simBoxes.has(beyond)) return '';
      simBoxes.delete(next);
      simBoxes.add(beyond);
    }
    simPlayer = next;
  }
  for (const b of simBoxes) if (level.goalSet[b] !== 1) return '';
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
