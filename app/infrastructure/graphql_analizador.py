"""Analiza una consulta GraphQL con `graphql-core` (el mismo analizador que usa el esquema real)."""

from graphql import GraphQLSyntaxError, OperationDefinitionNode, parse

from app.application.reportes import ConsultaInvalida


class AnalizadorGraphqlCore:
    def operaciones(self, consulta: str) -> list[str]:
        try:
            documento = parse(consulta)
        except GraphQLSyntaxError as e:
            raise ConsultaInvalida(f"La consulta no es GraphQL válido: {e.message}") from e
        return [d.operation.value for d in documento.definitions if isinstance(d, OperationDefinitionNode)]