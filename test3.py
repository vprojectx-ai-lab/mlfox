import mlfox

model = mlfox()
model.fit("Employee.csv", target="LeaveOrNot")

model.report()


