from dataclasses import dataclass


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    filename: str
    anatomy: str
    projection: str
    expected_fracture: bool

    @property
    def selector_label(self) -> str:
        return f"{self.case_id.replace('_', ' ').title()} · {self.anatomy}"


DEMO_CASES = (
    DemoCase(
        "caso_01",
        "09678e1b-distalUR_334257201609240543_front.png",
        "Radio/cúbito distal",
        "Frontal",
        True,
    ),
    DemoCase(
        "caso_02",
        "0e91fd4d-midshaftUR_295711201510240018_side.png",
        "Diáfisis de radio/cúbito",
        "Lateral",
        True,
    ),
    DemoCase(
        "caso_03",
        "310c070b-proximalUR_608317202211280374_side.png",
        "Radio/cúbito proximal",
        "Lateral",
        True,
    ),
    DemoCase(
        "caso_04",
        "SHF_001.jpg",
        "Húmero supracondíleo",
        "Radiografía de referencia",
        True,
    ),
    DemoCase(
        "caso_05",
        "UR_001.jpg",
        "Radio/cúbito",
        "Radiografía de referencia",
        True,
    ),
    DemoCase(
        "caso_06",
        "WRI_001.png",
        "Muñeca",
        "Radiografía de referencia",
        True,
    ),
    DemoCase(
        "caso_07",
        "12667489-proximalUR_389715201712230734_side.png",
        "Radio/cúbito proximal",
        "Lateral",
        False,
    ),
    DemoCase(
        "caso_08",
        "3796cf71-distalUR_605664202211070678_side.png",
        "Radio/cúbito distal",
        "Lateral",
        False,
    ),
    DemoCase(
        "caso_09",
        "431064ca-proximalUR_492231202001110685_front.png",
        "Radio/cúbito proximal",
        "Frontal",
        False,
    ),
    DemoCase(
        "caso_10",
        "NoF_UR_001.jpg",
        "Radio/cúbito",
        "Radiografía de referencia",
        False,
    ),
    DemoCase(
        "caso_11",
        "NoF_UR_002.jpg",
        "Radio/cúbito",
        "Radiografía de referencia",
        False,
    ),
    DemoCase(
        "caso_12",
        "NoF_UR_003.jpg",
        "Radio/cúbito",
        "Radiografía de referencia",
        False,
    ),
)

CASES_BY_ID = {case.case_id: case for case in DEMO_CASES}
DEFAULT_CASE_ID = "caso_04"
