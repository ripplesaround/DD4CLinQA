from torch.utils.data import RandomSampler


class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  def printname(self):
      print("hi")
      print(self.firstname, self.lastname)

# x = Student("Elon", "Musk")
# x.printname()
train_dataset_idx = range(100)
test1 = RandomSampler(train_dataset_idx)
test2 = RandomSampler(train_dataset_idx)
for i,item in enumerate(test1):
    print("%s",item)
# print("---")
# for i, item in enumerate(test2):
#     print("%s", item)