from sympy.logic.boolalg import truth_table
from sympy import symbols, Not, Or, And, simplify_logic, Implies, satisfiable
from itertools import product

# Zadanie 1
p, q = symbols('p q')
KB = Or(And(p, q), And(p, Not(q)))
alpha2 = And(True, True)

table_kb = list(truth_table(KB, [p, q]))
table_alpha2 = list(truth_table(alpha2, [p, q]))

is_safe = table_kb.count(True) >= table_alpha2.count(True)
print("Czy KB |= α2? ", is_safe)

# Zadanie 2
p, q = symbols('p q')
sentence1 = Not(Or(p, And(Not(p), q)))
sentence2 = And(Not(p), Not(q))

simplified_sentence1 = simplify_logic(sentence1)
simplified_sentence2 = simplify_logic(sentence2)

is_equivalent = simplified_sentence1 == simplified_sentence2
print("Czy zdania są logicznie równoważne? ", is_equivalent)

# Zadanie 3
p, q, r = symbols('p q r')

sentence3 = (p >> q) >> (Not(p) >> Not(q))
sentence4 = (p >> q) >> ((p & r) >> q)

is_satisfiable1 = satisfiable(sentence3)
is_satisfiable2 = satisfiable(sentence4)

print("Czy zdanie (i) jest spełnialne? ", is_satisfiable1)
print("Czy zdanie (ii) jest spełnialne? ", is_satisfiable2)

# Zadanie 4
p, q, r = symbols('p q r')

def implies_entails(p, q, r):
    table = list(product([False, True], repeat=3))
    for row in table:
        p_val, q_val, r_val = row
        if (p_val and not q_val) != ((p >> q).subs([(p, p_val), (q, q_val)]) and
                                     ((p & r) >> q).subs([(p, p_val), (q, q_val), (r, r_val)])):
            return False
    return True

entails = implies_entails(p, q, r)
print("Czy (p ⇒ q) |= ((p ∧ r) ⇒ q)? ", entails)

# Zadanie 5
cnf = simplify_logic(sentence3, form='cnf')
dnf = simplify_logic(sentence3, form='dnf')
print("CNF dla zdania (i): ", cnf)
print("DNF dla zdania (i): ", dnf)