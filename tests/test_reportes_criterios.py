"""G4 · criterios de selección, agrupación, orden y límite de un reporte, y varias tablas."""

import pytest

from app.domain.criterios import (
    MAX_CRITERIOS,
    leer_especificacion,
    normalizar_consulta,
)
from app.domain.errores import ReglaViolada
from app.domain.tablas import Tabla, aplicar_especificacion, tablas_desde_respuesta

MOROSOS = [
    {
        "codigo": "A-101",
        "bloque": "A",
        "deudaVencida": "135.00",
        "expensasVencidas": 1,
        "propietarioNombre": "Ana",
    },
    {
        "codigo": "B-101",
        "bloque": "B",
        "deudaVencida": "540.00",
        "expensasVencidas": 3,
        "propietarioNombre": "Beto",
    },
    {
        "codigo": "B-102",
        "bloque": "B",
        "deudaVencida": "1000.00",
        "expensasVencidas": 2,
        "propietarioNombre": "Carla",
    },
    {
        "codigo": "C-101",
        "bloque": "C",
        "deudaVencida": "900.00",
        "expensasVencidas": 4,
        "propietarioNombre": None,
    },
]
PAGOS = [
    {"monto": "300.00", "fechaPago": "2026-11-03", "estado": "PENDIENTE"},
    {"monto": "50.00", "fechaPago": "2026-11-05", "estado": "PENDIENTE"},
]


def _aplicar(especificacion: str, tablas: list[Tabla] | None = None) -> list[Tabla]:
    consulta = "{ morosos { codigo } }\n# ESPECIFICACION: " + especificacion
    return aplicar_especificacion(
        tablas or [Tabla("morosos", [dict(f) for f in MOROSOS])], leer_especificacion(consulta)
    )


def _codigos(tabla: Tabla) -> list[object]:
    return [f.get("codigo") for f in tabla.filas]


# ------------------------------------------------------------------ lectura y validación


def test_G4_sin_especificacion_el_reporte_sale_como_lo_devuelve_gestion() -> None:
    consulta = "{ morosos { codigo } }"
    assert leer_especificacion(consulta).vacia
    assert normalizar_consulta(consulta) == consulta
    tablas = [Tabla("morosos", MOROSOS)]
    assert aplicar_especificacion(tablas, leer_especificacion(consulta)) is tablas


def test_G4_la_especificacion_se_mueve_al_final_para_que_la_consulta_empiece_por_query_o_llave() -> None:
    consulta = '# ESPECIFICACION: {"limite": 2}\n{ morosos { codigo } }'

    normal = normalizar_consulta(consulta)

    assert normal.lstrip().startswith("{")
    assert normal.splitlines()[-1].startswith("# ESPECIFICACION:")
    assert leer_especificacion(normal).limite == 2


@pytest.mark.parametrize(
    "especificacion",
    [
        "no es json",
        "[1, 2]",
        '{"ejecutar": "rm -rf"}',
        '{"filtros": [{"campo": "bloque", "op": "LIKE", "valor": "B"}]}',
        '{"filtros": [{"campo": "bloque", "op": "="}]}',
        '{"filtros": [{"campo": "bloque; drop", "op": "=", "valor": "B"}]}',
        '{"filtros": [{"campo": "bloque", "op": "=", "valor": {"$gt": ""}}]}',
        '{"filtros": [{"campo": "bloque", "op": "=", "valor": "B", "extra": 1}]}',
        '{"limite": 0}',
        '{"limite": 10001}',
        '{"limite": true}',
        '{"limite": "5"}',
        '{"orden": [{"campo": "deudaVencida", "sentido": "arriba"}]}',
        '{"agrupar": {}}',
        '{"agrupar": {"por": ["bloque"], "calcular": [{"fn": "eval", "campo": "x"}]}}',
        '{"agrupar": {"por": ["bloque"], "calcular": [{"fn": "suma"}]}}',
        '{"agrupar": {"calcular": [{"fn": "contar", "como": "a=b"}]}}',
        '{"ocultar": "bloque"}',
    ],
)
def test_G4_una_especificacion_mal_formada_se_rechaza_con_el_codigo_de_la_regla(especificacion: str) -> None:
    with pytest.raises(ReglaViolada) as e:
        leer_especificacion("{ x }\n# ESPECIFICACION: " + especificacion)

    assert e.value.codigo == "G4"


def test_G4_mas_de_una_linea_o_una_especificacion_enorme_se_rechazan() -> None:
    with pytest.raises(ReglaViolada, match="más de una"):
        leer_especificacion("{ x }\n# ESPECIFICACION: {}\n# ESPECIFICACION: {}")
    with pytest.raises(ReglaViolada, match="supera"):
        leer_especificacion('{ x }\n# ESPECIFICACION: {"filtros": [], "ocultar": ["' + "a" * 5000 + '"]}')


def test_G4_hay_un_tope_de_criterios() -> None:
    filtros = ",".join(['{"campo": "bloque", "op": "=", "valor": "B"}'] * (MAX_CRITERIOS + 1))

    with pytest.raises(ReglaViolada, match="filtros"):
        leer_especificacion('{ x }\n# ESPECIFICACION: {"filtros": [' + filtros + "]}")


# ------------------------------------------------------------------ filtros


def test_G4_el_texto_se_compara_sin_distinguir_mayusculas() -> None:
    (tabla,) = _aplicar('{"filtros": [{"campo": "bloque", "op": "=", "valor": "b"}]}')

    assert _codigos(tabla) == ["B-101", "B-102"]


def test_G4_los_numeros_se_comparan_como_numeros_no_como_texto() -> None:
    (tabla,) = _aplicar('{"filtros": [{"campo": "deudaVencida", "op": ">", "valor": 900}]}')

    assert _codigos(tabla) == ["B-102"]  # "1000.00" > 900 aunque como texto "1000.00" < "900"


def test_G4_varios_criterios_se_combinan_con_y() -> None:
    (tabla,) = _aplicar(
        '{"filtros": [{"campo": "bloque", "op": "=", "valor": "B"}, '
        '{"campo": "expensasVencidas", "op": ">=", "valor": 3}]}'
    )

    assert _codigos(tabla) == ["B-101"]


def test_G4_contiene_distinto_y_los_datos_nulos() -> None:
    (con,) = _aplicar('{"filtros": [{"campo": "propietarioNombre", "op": "contiene", "valor": "ar"}]}')
    (distinto,) = _aplicar('{"filtros": [{"campo": "propietarioNombre", "op": "!=", "valor": "Ana"}]}')

    assert _codigos(con) == ["B-102"]  # Carla
    assert _codigos(distinto) == ["B-101", "B-102", "C-101"]  # el nulo no es «Ana»


def test_G4_las_fechas_iso_se_comparan_por_orden() -> None:
    (tabla,) = _aplicar(
        '{"filtros": [{"campo": "fechaPago", "op": ">=", "valor": "2026-11-04"}]}',
        [Tabla("pagosPendientes", PAGOS)],
    )

    assert [f["monto"] for f in tabla.filas] == ["50.00"]


# ------------------------------------------------------------------ agrupar, ordenar, limitar, ocultar


def test_G4_agrupar_por_bloque_suma_la_deuda_y_cuenta_las_unidades_con_dinero_como_texto_decimal() -> None:
    especificacion = (
        '{"agrupar": {"por": ["bloque"], '
        '"calcular": [{"fn": "suma", "campo": "deudaVencida", "como": "Deuda total"},'
        ' {"fn": "contar", "como": "Unidades"}]}}'
    )

    (tabla,) = _aplicar(especificacion)

    assert tabla.filas == [
        {"bloque": "A", "Deuda total": "135.00", "Unidades": 1},
        {"bloque": "B", "Deuda total": "1540.00", "Unidades": 2},
        {"bloque": "C", "Deuda total": "900.00", "Unidades": 1},
    ]


def test_G4_promedio_minimo_y_maximo_y_totales_sin_agrupar() -> None:
    especificacion = (
        '{"agrupar": {"calcular": [{"fn": "promedio", "campo": "deudaVencida", "como": "Promedio"},'
        ' {"fn": "minimo", "campo": "deudaVencida", "como": "Menor"},'
        ' {"fn": "maximo", "campo": "deudaVencida", "como": "Mayor"}]}}'
    )

    (tabla,) = _aplicar(especificacion)

    assert tabla.filas == [{"Promedio": "643.75", "Menor": "135.00", "Mayor": "1000.00"}]


def test_G4_ordenar_numericamente_y_despues_de_agrupar_por_el_nombre_del_calculo() -> None:
    (por_deuda,) = _aplicar('{"orden": [{"campo": "deudaVencida", "sentido": "desc"}]}')
    (por_total,) = _aplicar(
        '{"agrupar": {"por": ["bloque"], '
        '"calcular": [{"fn": "suma", "campo": "deudaVencida", "como": "Total"}]},'
        ' "orden": [{"campo": "Total", "sentido": "desc"}], "limite": 1}'
    )

    assert _codigos(por_deuda) == ["B-102", "C-101", "B-101", "A-101"]  # 1000 > 900 > 540 > 135
    assert por_total.filas == [{"bloque": "B", "Total": "1540.00"}]


def test_G4_los_datos_nulos_quedan_al_final_al_ordenar() -> None:
    (tabla,) = _aplicar('{"orden": [{"campo": "propietarioNombre"}]}')

    assert _codigos(tabla) == ["A-101", "B-101", "B-102", "C-101"]
    assert tabla.filas[-1]["propietarioNombre"] is None


def test_G4_limite_y_columnas_auxiliares_ocultas() -> None:
    (tabla,) = _aplicar(
        '{"filtros": [{"campo": "bloque", "op": "=", "valor": "B"}], '
        '"limite": 1, "ocultar": ["bloque", "expensasVencidas"]}'
    )

    assert tabla.filas == [{"codigo": "B-101", "deudaVencida": "540.00", "propietarioNombre": "Beto"}]


# ------------------------------------------------------------------ un criterio nunca se ignora en silencio


def test_G4_un_campo_que_el_resultado_no_tiene_se_rechaza_en_vez_de_ignorarse() -> None:
    with pytest.raises(ReglaViolada, match="no está en el resultado"):
        _aplicar('{"filtros": [{"campo": "piso", "op": "=", "valor": "3"}]}')


def test_G4_sumar_un_campo_no_numerico_se_rechaza() -> None:
    with pytest.raises(ReglaViolada, match="no es numérico"):
        _aplicar(
            '{"agrupar": {"por": ["bloque"], "calcular": [{"fn": "suma", "campo": "propietarioNombre"}]}}'
        )


def test_G4_una_tabla_que_no_existe_se_rechaza_nombrando_las_que_hay() -> None:
    with pytest.raises(ReglaViolada, match="morosos"):
        _aplicar('{"tabla": "reservas", "limite": 3}')


def test_G4_un_resultado_vacio_no_hace_fallar_los_criterios() -> None:
    (tabla,) = _aplicar('{"filtros": [{"campo": "bloque", "op": "=", "valor": "B"}]}', [Tabla("morosos", [])])

    assert tabla.filas == []


# ------------------------------------------------------------------ varias tablas


def test_G4_cada_lista_de_la_consulta_es_una_tabla() -> None:
    datos = {"morosos": MOROSOS, "pagosPendientes": PAGOS, "salud": {"estado": "UP"}}

    tablas = tablas_desde_respuesta(datos)

    assert [t.nombre for t in tablas] == ["morosos", "pagosPendientes"]
    assert len(tablas[1].filas) == 2


def test_G4_sin_listas_el_resultado_es_una_sola_fila_como_antes() -> None:
    (tabla,) = tablas_desde_respuesta({"estadoDeCuenta": {"codigo": "A-101", "deudaTotal": "10.00"}})

    assert tabla.filas == [{"estadoDeCuenta.codigo": "A-101", "estadoDeCuenta.deudaTotal": "10.00"}]


def test_G4_un_criterio_sin_tabla_solo_afecta_a_las_que_tienen_ese_campo() -> None:
    tablas = [
        Tabla("morosos", [dict(f) for f in MOROSOS]),
        Tabla("pagosPendientes", [dict(f) for f in PAGOS]),
    ]

    morosos, pagos = _aplicar('{"filtros": [{"campo": "bloque", "op": "=", "valor": "B"}]}', tablas)

    assert _codigos(morosos) == ["B-101", "B-102"]
    assert len(pagos.filas) == 2  # no tiene «bloque»: queda como estaba


def test_G4_con_tabla_explicita_el_criterio_solo_toca_esa_y_exige_el_campo() -> None:
    tablas = [
        Tabla("morosos", [dict(f) for f in MOROSOS]),
        Tabla("pagosPendientes", [dict(f) for f in PAGOS]),
    ]

    morosos, pagos = _aplicar('{"tabla": "pagosPendientes", "limite": 1}', tablas)

    assert len(morosos.filas) == 4 and len(pagos.filas) == 1
    with pytest.raises(ReglaViolada):
        _aplicar(
            '{"tabla": "pagosPendientes", "filtros": [{"campo": "bloque", "op": "=", "valor": "B"}]}', tablas
        )


# ------------------------------------------------------------------ fallos reales del LLM (producción, 08/11)


def test_G4_el_comentario_en_la_misma_linea_que_la_consulta_tambien_se_lee() -> None:
    """Groq escribió la consulta y el comentario en una sola línea y el «5 primeros» se ignoró en silencio."""
    consulta = (
        "{ morosos { codigo deudaVencida } } # ESPECIFICACION: "
        '{"orden": [{"campo": "deudaVencida", "sentido": "desc"}], "limite": 2}'
    )

    especificacion = leer_especificacion(consulta)
    (tabla,) = aplicar_especificacion([Tabla("morosos", [dict(f) for f in MOROSOS])], especificacion)

    assert especificacion.limite == 2
    assert _codigos(tabla) == ["B-102", "C-101"]
    normal = normalizar_consulta(consulta)
    assert normal.startswith("{ morosos { codigo deudaVencida } }")
    assert normal.splitlines()[-1].startswith("# ESPECIFICACION:")
    assert leer_especificacion(normal).limite == 2


def test_G4_con_tilde_o_en_minuscula_el_comentario_tambien_se_lee() -> None:
    assert leer_especificacion('{ x }\n# Especificación: {"limite": 3}').limite == 3


@pytest.mark.parametrize(
    "consulta",
    [
        '{ x }\n# ESPECIFICACION {"limite": 3}',  # sin los dos puntos
        "{ x }\n# ESPECIFICACION:",  # sin contenido
        '{ x }\n# CRITERIOS: {"limite": 3}\n# especificacion de criterios',  # la menciona pero no la declara
    ],
)
def test_G4_si_el_llm_menciona_la_especificacion_pero_la_escribe_mal_se_rechaza_no_se_ignora(
    consulta: str,
) -> None:
    with pytest.raises(ReglaViolada) as e:
        leer_especificacion(consulta)

    assert e.value.codigo == "G4"
