"""Adaptadores en memoria de los puertos de rostros: para pruebas y desarrollo (no sobreviven al reinicio)."""

from uuid import UUID, uuid4

from app.domain.rostros import Desconocido, PerfilFacial, Prediccion


class RepositorioPerfilesMemoria:
    def __init__(self) -> None:
        self._perfiles: dict[UUID, PerfilFacial] = {}

    def guardar(self, perfil: PerfilFacial) -> None:
        self._perfiles[perfil.residente_id] = perfil

    def por_residente(self, residente_id: UUID) -> PerfilFacial | None:
        return self._perfiles.get(residente_id)

    def con_consentimiento_vigente(self) -> list[PerfilFacial]:
        return [p for p in self._perfiles.values() if p.vigente]


class RepositorioDesconocidosMemoria:
    def __init__(self) -> None:
        self.guardados: list[Desconocido] = []

    def guardar(self, desconocido: Desconocido) -> None:
        self.guardados.append(desconocido)


class RepositorioPrediccionesMemoria:
    def __init__(self) -> None:
        self.guardadas: list[tuple[str, Prediccion]] = []

    def guardar(self, prediccion: Prediccion) -> str:
        prediccion_id = str(uuid4())
        self.guardadas.append((prediccion_id, prediccion))
        return prediccion_id


class AlmacenFotosMemoria:
    def __init__(self) -> None:
        self.objetos: dict[str, bytes] = {}
        self.borrados = 0

    def guardar(self, residente_id: UUID, fotos: list[bytes]) -> list[str]:
        claves = [f"rostros/{residente_id}/{uuid4()}.jpg" for _ in fotos]
        self.objetos.update(zip(claves, fotos, strict=True))
        return claves

    def borrar(self, claves: list[str]) -> None:
        self.borrados += 1
        for clave in claves:
            self.objetos.pop(clave, None)
