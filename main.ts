// Room-by-room push-optimal Sokoban solver.
//
// The levels this competition draws are a hallway with one to four rooms off
// it, each room holding 3 or 4 boxes and as many goals, and a box pushed out
// of a room's door into the hallway can never be pushed back -- so every box
// stays in its room and a level's push-optimal count is the sum of its
// rooms' optima. This entry exploits that structure directly:
//
//   1. It cuts the level into rooms: squares of the player's (hallway) row
//      from which no box can ever reach a goal are dead, and cutting them
//      out of the floor leaves each room as its own connected part.
//   2. It solves each room push-optimally, in turn, with A* over pushes:
//      a state is the boxes' squares plus the player's reachable region
//      (named by its lowest square), each push costs 1, and the heuristic
//      is the cheapest assignment of boxes to goals by shortest-path
//      distance, which never overestimates the pushes left, so the first
//      solved state popped is push-optimal. Dead squares (a box there
//      reaches no goal, by pulling backwards from the goals) and frozen
//      2x2 blocks prune states no push can save. If A* spends its state
//      budget, a plain breadth-first search over pushes takes over.
//   3. It stitches the rooms together: the player walks on from where the
//      last room left it, other rooms' boxes standing as walls (on their
//      goals once their room is done), so the moves of one room never
//      disturb another.
//
// Determinism: every budget is counted in states expanded, never in time,
// and every tie is broken by square order and insertion order, so two runs
// on the same set answer byte for byte alike. The program reads no clock.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, in states expanded: A* per room, its breadth-first fallback,
// and the whole run across all five instances. Work, never the clock.
const ASTAR_STATES = 250_000;
const BFS_STATES = 500_000;
const TOTAL_STATES = 6_000_000;

// The four steps: [dx, dy, the walk letter, the push letter].
const DIRS: readonly (readonly [number, number, string, string])[] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: its width and height, walls, goals, boxes, player. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] as string;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board,
  // and no neighbour lookup ever runs past the array's ends.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The squares from which a box alone can be pushed onto some goal: pull
 * backwards from the goals over open floor. A square a pull never reaches
 * is dead -- a box there reaches no goal -- and any state holding one is
 * pruned. (A pull from square x to square y = x - d needs y open and the
 * square beyond it, y - d, open for the player to stand on.)
 */
function liveSquares(board: Board): Uint8Array {
  const { w, wall, goals } = board;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi] as number;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Every floor square's walking distance from `from`, around walls alone. */
function distances(board: Board, from: number): Int32Array {
  const { w, wall } = board;
  const dist = new Int32Array(wall.length).fill(-1);
  const queue: number[] = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi] as number;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (!wall[q] && dist[q] < 0) {
        dist[q] = dist[p] as number + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room, with its squares, boxes and goals. Null when the
 * level is no set of rooms -- a box starts in the hallway, a goal lies in
 * the cut, or some part holds more boxes than goals or fewer.
 */
function roomsOf(board: Board): { squares: number[]; boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = board;
  const live = liveSquares(board);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = (squares[i] as number) + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) (rooms[part[b] as number] as { boxes: number[] }).boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    (rooms[part[g] as number] as { goals: number[] }).goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** A push search's shared state: the board, its goals and dead squares, and the run's work budget. */
class Search {
  readonly w: number;
  readonly size: number;
  readonly wall: Uint8Array;
  readonly goal: Uint8Array;
  readonly live: Uint8Array;
  readonly goalDist: Int32Array[];
  private readonly seen: Int32Array;
  private readonly stack: Int32Array;
  private stamp = 0;
  spent = 0;
  private readonly budget: { total: number };

  constructor(board: Board, budget: { total: number }) {
    this.w = board.w;
    this.size = board.wall.length;
    this.wall = board.wall;
    this.goal = new Uint8Array(this.size);
    for (const g of board.goals) this.goal[g] = 1;
    this.live = liveSquares(board);
    this.goalDist = board.goals.map((g) => distances(board, g));
    this.seen = new Int32Array(this.size);
    this.stack = new Int32Array(this.size);
    this.budget = budget;
  }

  /** The player's reachable region from `from` around walls and `occupied`, marked with a fresh stamp; its lowest square. */
  regionLow(occupied: Uint8Array, from: number): number {
    const { w, wall, seen, stack } = this;
    const here = ++this.stamp;
    let low = from;
    let n = 0;
    stack[n++] = from;
    seen[from] = here;
    while (n > 0) {
      const p = stack[--n] as number;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
        if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== here) {
          seen[q] = here;
          stack[n++] = q;
        }
      }
    }
    return low;
  }

  /** Whether `cell` lies in the region the last regionLow call marked. */
  inRegion(cell: number): boolean {
    return this.seen[cell] === this.stamp;
  }

  /** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
  frozen(occupied: Uint8Array, t: number): boolean {
    const { w, wall, goal } = this;
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
    }
    return false;
  }

  /**
   * The cheapest assignment of `boxes` to goals by shortest-path distance
   * (every permutation, at most 4 boxes): each push moves one box one
   * square, so no solution from here costs fewer pushes than this -- an
   * admissible heuristic. Infinity when some box reaches no goal.
   */
  heuristic(boxes: number[]): number {
    const n = boxes.length;
    const order = boxes.map((_, i) => i);
    let best = Infinity;
    const permute = (k: number): void => {
      if (k === n) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = (this.goalDist[order[i] as number] as Int32Array)[boxes[i] as number] as number;
          if (d < 0) return;
          sum += d;
          if (sum >= best) return;
        }
        best = sum;
        return;
      }
      for (let i = k; i < n; i++) {
        [order[k], order[i]] = [order[i] as number, order[k] as number];
        permute(k + 1);
        [order[k], order[i]] = [order[i] as number, order[k] as number];
      }
    };
    permute(0);
    return best;
  }

  /** A state's key: the region's lowest square and the boxes' squares. */
  keyOf(boxes: number[], low: number): number | string {
    if (boxes.length <= 5) {
      let k = low;
      for (const b of boxes) k = k * this.size + b;
      return k;
    }
    return `${low}:${boxes.join(',')}`;
  }

  solved(boxes: number[]): boolean {
    return boxes.every((b) => this.goal[b]);
  }

  /** One more state spent: false when the run's total budget is gone. */
  charge(): boolean {
    this.spent++;
    this.budget.total++;
    return this.budget.total <= TOTAL_STATES;
  }
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(board: Board, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = board;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi] as number;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p] as number) s = (how[p] as string) + s;
  return s;
}

/**
 * A push-optimal search over `board`: A* over pushes with the assignment
 * heuristic, falling back to breadth-first search over pushes when A*
 * spends its state budget. Returns {pushes, moves} or null when both give
 * up (or the board has no solution). Pure: no clock, no randomness.
 */
function solvePushOptimal(board: Board, budget: { total: number }): { pushes: number; moves: string } | null {
  const search = new Search(board, budget);
  const { w } = board;
  if (board.boxes.some((b) => !search.live[b])) return null;
  if (search.solved(board.boxes)) return { pushes: 0, moves: '' };

  // The states in the order found: their boxes, the player's square after
  // the push that made them, their pushes so far, their key, and their parent.
  const statesBoxes: number[][] = [board.boxes.slice().sort((a, b) => a - b)];
  const statesPlayer: number[] = [board.player];
  const statesG: number[] = [0];
  const statesKey: (number | string)[] = [];
  const parent: number[] = [-1];
  const occupied = new Uint8Array(search.size);
  for (const b of board.boxes) occupied[b] = 1;
  const startLow = search.regionLow(occupied, board.player);
  for (const b of board.boxes) occupied[b] = 0;
  statesKey.push(search.keyOf(statesBoxes[0] as number[], startLow));
  const best = new Map<number | string, number>([[statesKey[0] as number | string, 0]]);

  // Every push a state offers: [box index, direction offset, target square].
  const pushesOf = (boxes: number[], player: number): [number, number, number][] => {
    for (const b of boxes) occupied[b] = 1;
    search.regionLow(occupied, player);
    const out: [number, number, number][] = [];
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi] as number;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (!search.inRegion(b - d) || search.wall[t] || occupied[t] || !search.live[t]) continue;
        out.push([bi, d, t]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
    return out;
  };

  // The child a push makes, or null when it is frozen or already known at
  // no more pushes: [boxes, player square, key].
  const childOf = (boxes: number[], bi: number, t: number): { boxes: number[]; player: number; key: number | string } | null => {
    const b = boxes[bi] as number;
    for (const x of boxes) occupied[x] = 1;
    occupied[b] = 0;
    occupied[t] = 1;
    const frozen = search.frozen(occupied, t);
    let out: { boxes: number[]; player: number; key: number | string } | null = null;
    if (!frozen) {
      const next = boxes.slice();
      next[bi] = t;
      next.sort((a, b2) => a - b2);
      const low = search.regionLow(occupied, b);
      out = { boxes: next, player: b, key: search.keyOf(next, low) };
    }
    // Clear the target square again: it is not one of the parent's boxes,
    // so the loop below would leave a phantom box standing on it.
    occupied[t] = 0;
    for (const x of boxes) occupied[x] = 0;
    return out;
  };

  // Rebuild the moves from a solved state's parent chain: walk to each
  // push's square, then push.
  const movesOf = (found: number): string => {
    const path: number[] = [];
    for (let s = found; s > 0; s = parent[s] as number) path.push(s);
    path.reverse();
    let moves = '';
    let player = board.player;
    let boxes = board.boxes.slice().sort((a, b) => a - b);
    for (const s of path) {
      const nextBoxes = statesBoxes[s] as number[];
      const from = boxes.find((b) => !nextBoxes.includes(b)) as number;
      const to = nextBoxes.find((b) => !boxes.includes(b)) as number;
      const d = to - from;
      const dir = DIRS.find(([dx, dy]) => dy * w + dx === d) as readonly [number, number, string, string];
      for (const b of boxes) occupied[b] = 1;
      moves += walkTo(board, occupied, player, from - d);
      for (const b of boxes) occupied[b] = 0;
      moves += dir[3];
      player = from;
      boxes = nextBoxes;
    }
    return moves;
  };

  // --- A* over pushes. The heap orders by (f, deeper first, found first). ---
  const heap: number[] = [0];
  const fScore: number[] = [search.heuristic(statesBoxes[0] as number[])];
  const heapLess = (a: number, b: number): boolean => {
    const fa = (fScore[a] as number);
    const fb = (fScore[b] as number);
    if (fa !== fb) return fa < fb;
    if (statesG[a] !== statesG[b]) return (statesG[a] as number) > (statesG[b] as number);
    return a < b;
  };
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i] as number, heap[p] as number)) break;
      [heap[i], heap[p]] = [heap[p] as number, heap[i] as number];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0] as number;
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1] as number, heap[c] as number)) c++;
        if (!heapLess(heap[c] as number, heap[i] as number)) break;
        [heap[i], heap[c]] = [heap[c] as number, heap[i] as number];
        i = c;
      }
    }
    return top;
  };

  let found = -1;
  let expanded = 0;
  while (heap.length > 0 && expanded < ASTAR_STATES) {
    const s = heapPop();
    const boxes = statesBoxes[s] as number[];
    if (best.get(statesKey[s] as number | string) !== statesG[s]) {
      // Stale heap entry: a better path to this state was found since.
      continue;
    }
    if (search.solved(boxes)) {
      found = s;
      break;
    }
    expanded++;
    if (!search.charge()) return null;
    const g = statesG[s] as number;
    for (const [bi, , t] of pushesOf(boxes, statesPlayer[s] as number)) {
      const child = childOf(boxes, bi, t);
      if (child === null) continue;
      const seenG = best.get(child.key);
      if (seenG !== undefined && seenG <= g + 1) continue;
      const h = search.heuristic(child.boxes);
      if (!Number.isFinite(h)) continue;
      const t2 = statesBoxes.length;
      statesBoxes.push(child.boxes);
      statesPlayer.push(child.player);
      statesG.push(g + 1);
      statesKey.push(child.key);
      parent.push(s);
      fScore.push(g + 1 + h);
      best.set(child.key, g + 1);
      heapPush(t2);
    }
  }
  if (found >= 0) return { pushes: statesG[found] as number, moves: movesOf(found) };
  if (heap.length === 0) return null; // A* proved there is no solution.
  if (budget.total > TOTAL_STATES) return null; // The run's budget is gone.

  // --- Breadth-first fallback over pushes, from scratch. ---
  const bfsBoxes: number[][] = [board.boxes.slice().sort((a, b) => a - b)];
  const bfsPlayer: number[] = [board.player];
  const bfsParent: number[] = [-1];
  const bfsSeen = new Set<number | string>();
  for (const b of board.boxes) occupied[b] = 1;
  bfsSeen.add(search.keyOf(bfsBoxes[0] as number[], search.regionLow(occupied, board.player)));
  for (const b of board.boxes) occupied[b] = 0;
  let bfsFound = -1;
  for (let head = 0; bfsFound < 0 && head < bfsBoxes.length && head < BFS_STATES; head++) {
    const boxes = bfsBoxes[head] as number[];
    if (!search.charge()) return null;
    for (const [bi, , t] of pushesOf(boxes, bfsPlayer[head] as number)) {
      const child = childOf(boxes, bi, t);
      if (child === null || bfsSeen.has(child.key)) continue;
      bfsSeen.add(child.key);
      bfsBoxes.push(child.boxes);
      bfsPlayer.push(child.player);
      bfsParent.push(head);
      if (search.solved(child.boxes)) {
        bfsFound = bfsBoxes.length - 1;
        break;
      }
    }
  }
  if (bfsFound < 0) return null;
  // Rebuild the moves from the breadth-first chain (same walk-and-push walk).
  const bfsPath: number[] = [];
  for (let s = bfsFound; s > 0; s = bfsParent[s] as number) bfsPath.push(s);
  bfsPath.reverse();
  let moves = '';
  let player = board.player;
  let boxes = board.boxes.slice().sort((a, b) => a - b);
  for (const s of bfsPath) {
    const nextBoxes = bfsBoxes[s] as number[];
    const from = boxes.find((b) => !nextBoxes.includes(b)) as number;
    const to = nextBoxes.find((b) => !boxes.includes(b)) as number;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d) as readonly [number, number, string, string];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(board, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: bfsPath.length, moves };
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. When a
 * room's search gives up, the moves so far are returned (a valid answer,
 * worth nothing); when the level is no set of rooms, the whole level is
 * searched at once.
 */
function solveLevel(board: Board, budget: { total: number }): string {
  const rooms = roomsOf(board);
  if (rooms === null) {
    const whole = solvePushOptimal(board, budget);
    return whole === null ? '' : whole.moves;
  }
  let player = board.player;
  let moves = '';
  const done = new Set<number>();
  for (let r = 0; r < rooms.length; r++) {
    const room = rooms[r] as { squares: number[]; boxes: number[]; goals: number[] };
    const wall = board.wall.slice();
    rooms.forEach((other, oi) => {
      if (oi === r) return;
      for (const b of done.has(oi) ? other.goals : other.boxes) wall[b] = 1;
    });
    const sub: Board = {
      w: board.w,
      h: board.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushOptimal(sub, budget);
    if (solved === null) return moves;
    moves += solved.moves;
    player = endOf(board.w, player, solved.moves);
    done.add(r);
  }
  return moves;
}

const budget = { total: 0 };
for (const id of readdirSync(IN_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(parseLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8')), budget);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
