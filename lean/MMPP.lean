import Mathlib.Tactic

/-!
Two finite certificates from the paper, checked by kernel evaluation.

1. `cert12`: the twelve-state cycle at t = 19/2.  With B = I + C/r written as Bint / M,
   M = 1001500, a = r t = 38057/4000, N = 80, the inequality
       300000 (S + w1 R) + E + R < 0
   of Section 4 (S the truncated signed series, E the truncated exponential series, R the
   tail bound, w1 the l1 norm of pi - alpha) is multiplied by the positive integer
       K = 12000 * 4000^(N+1) * (N+1)! * M^N * (4000 (N+2) - 38057)
   and stated over the integers.  It implies P(T^pi > 19/2) - P(T^alpha > 19/2) < -1/300000.

2. `six_cycle_second_derivative`: for the unit directed six-cycle and
   d = (1/3, 2/5, 2/3, 1, 2/3, 2/5),  2 pi D^-1 Q D^-1 Q D^-1 1 = -1.
-/

namespace MMPP

def ps : List Int := [1500, 1433, 1250, 1000, 750, 567, 500, 567, 750, 1000, 1250, 1433]
def cs : List Int := ps.map (fun p => 1500 - p)     -- diagonal of Bint
def ws : List Int := ps.map (fun p => 1000 - p)     -- 12000 (pi - alpha)
def W : Int := (ws.map Int.natAbs).foldl (fun s x => s + (x : Int)) 0

/-- the successor vector (v_{i+1})_i on the cycle -/
def rot (v : List Int) : List Int := v.tail ++ [v.headD 0]
/-- one step of the integer recurrence v_{k+1} = Bint v_k -/
def step (v : List Int) : List Int :=
  List.zipWith (· + ·) (List.zipWith (· * ·) cs v) ((rot v).map (fun x => 1000000 * x))
def dot (a b : List Int) : Int := (List.zipWith (· * ·) a b).foldl (· + ·) 0

def N : Nat := 80
def A : Int := 38057
def M : Int := 1001500

/-- (k+1)(k+2)...(N+1) = (N+1)!/k! -/
def ff (k : Nat) : Int := ((List.range (N + 1 - k)).map (fun j => ((k + 1 + j : Nat) : Int))).foldl (· * ·) 1

def term (k : Nat) (v : List Int) : Int :=
  (300000 * A ^ k * dot ws v * M ^ (N - k) + 12000 * A ^ k * M ^ N) * 4000 ^ (N + 1 - k) * ff k
    * (4000 * ((N : Int) + 2) - A)

def loop : Nat → Nat → List Int → Int → Int
  | 0, _, _, acc => acc
  | f + 1, k, v, acc => loop f (k + 1) (step v) (acc + term k v)

def lhs : Int :=
  loop (N + 1) 0 (List.replicate 12 1) 0 + (300000 * W + 12000) * A ^ (N + 1) * M ^ N * 4000 * ((N : Int) + 2)

theorem W_eq : W = 3732 := by decide +kernel

theorem cert12 : lhs < 0 := by decide +kernel

/-- the six-cycle second derivative -/
def d6 : List ℚ := [1/3, 2/5, 2/3, 1, 2/3, 2/5]
def rotq (v : List ℚ) : List ℚ := v.tail ++ [v.headD 0]
def Qv (v : List ℚ) : List ℚ := List.zipWith (· - ·) (rotq v) v
def Dinv (v : List ℚ) : List ℚ := List.zipWith (· / ·) v d6
def sumq (v : List ℚ) : ℚ := v.foldl (· + ·) 0

theorem six_cycle_second_derivative :
    2 * sumq (Dinv (Qv (Dinv (Qv (Dinv (List.replicate 6 1)))))) / 6 = -1 := by
  decide +kernel

end MMPP
