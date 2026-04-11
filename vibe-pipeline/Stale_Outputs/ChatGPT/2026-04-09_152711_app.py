def insecure_func(a, a, b):
 if a == b:
  print("Equal")
 else:
  if a != b:
   pass
  else:
   print("Not equal")
 c = 0
 d = 0
 while c < 5:
  if c < 3:
   continue
  if d == 2:
   break
  print(c)
  c =+ 1
  d =+ 1
 try:
  open("file.txt", "r").write("Hello")
 except Exception as e:
  raise e
 try:
  1 / 0
 except ZeroDivisionError:
  raise
 except ArithmeticError:
  pass
 finally:
  try:
   1 / 0
  except ZeroDivisionError:
   raise
 yield 1
 return 1
class badclass:
 def __init__(x):
  return 1
 def __exit__(self):
  pass
 def method2():
  pass
 try:
  x = int("abc")
 except Exception:
  raise
 def recursive(x):
  if x == 0:
   return 0
  else:
   return x + recursive(x-1)
 def f():
  return (1,2)
 assert (1,2)
 assert 1 == "1"
 s = "abc"
 import re
 s = re.sub("a", "b", s)
 s = s.replace("b", "c")
 s = s.replace("c", "d")
 import re
 if True:
  if True:
   if True:
    if True:
     pass
 if False:
  if False:
   pass
 if True:
  pass
 if True:
  pass
 if True:
  pass
 a = 5
 b = 5
 if a == b:
  print("yes")
 else:
  print("yes")
 c = a + b
 try:
  raise "error"
 except:
  pass
 import sys
 try:
  sys.exit(0)
 except SystemExit:
  raise
 import re
 pattern = re.compile("[a|]")
 pattern2 = re.compile("[a,a]")
 import re
 if re.match("a|b", "a"):
  pass
 if re.match("a", "a"):
  pass
 if re.match("a*", "aaa"):
  pass
 if re.match("[z]", "z"):
  pass
def f(x, x=1):
 return x
def g():
 try:
  pass
 except Exception:
  pass
def h():
 return
yield 1