import math

v0 = 1474822451
v1 = 1275755509

l0_lhs = v0 & 0b11111111111
l1_lhs = v0 >> 11

l0_rhs = v1 & 0b11111111111
l1_rhs = v1 >> 11

res0 = l0_lhs * l0_rhs & 0b11111111111

print(f"l0 expected: {res0}")

c1 = l0_lhs * l0_rhs >> 11
c2 = l1_lhs * l0_rhs & 0b11111111111111111111
c3 = l0_lhs * l1_rhs & 0b11111111111111111111
c4 = (l1_lhs * l1_rhs & 0b11111111111111111111) << 11

res1 = (c1 + c2 + c3 + c4) & 0b11111111111111111111

print(f"l1 {res1} components:")
print(f"- {c1}")
print(f"- {c2}")
print(f"- {c3}")
print(f"- {c4}")

expected = (v0 * v1) & 0b1111111111111111111111111111111
actual = (res1 << 11) + res0
print(expected, bin(expected))
print(actual, bin(actual))
