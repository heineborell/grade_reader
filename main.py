from getGrade import getGradeStudentId
from sheetmerger import merge_grade


def main():
    csv_paths = [
        "gc_2025CMN17.20BS105a01_fullgc_2025-10-19-16-43-45.csv",
        "gc_2025CMN17.20BS105a02_fullgc_2025-10-21-13-45-52.csv",
        "gc_2025CMN17.20BS105a03_fullgc_2025-10-21-14-15-07.csv",
    ]
    for student_no, grade in getGradeStudentId(static=False):
        print("Detected:", student_no, grade)
        if student_no and grade is not None:
            for i, path in enumerate(csv_paths):
                merge_grade(
                    path,
                    i,
                    (student_no, grade),
                )


if __name__ == "__main__":
    main()
