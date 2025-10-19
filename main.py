from getGrade import getGradeStudentId
from sheetmerger import merge_grade


def main():
    for student_no, grade in getGradeStudentId(static=False):
        print("Detected:", student_no, grade)
        merge_grade(
            "gc_2025CMN17.20BS105a01_fullgc_2025-10-19-16-43-45.csv",
            (student_no, grade),
        )


if __name__ == "__main__":
    main()
