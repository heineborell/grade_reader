from getGrade import getGradeStudentId


def main():
    for student_no, grade in getGradeStudentId(static=True):
        print("Detected:", student_no, grade)


if __name__ == "__main__":
    main()
