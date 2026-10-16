"""Errores del dominio. Dominio puro: sin FastAPI ni librerías de ML."""


class ReglaViolada(Exception):
    """Una regla de negocio lo impide. `codigo` es el de `reglas-de-negocio.md` (ej. `I1`)."""

    def __init__(self, codigo: str, mensaje: str) -> None:
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje
