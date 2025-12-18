from util import *


class FOL_Concept:
    def __init__(self, name):
        self.name = name

    @property
    def name_with_article(self):
        if self.name[0] in ['a', 'e', 'i', 'o', 'u']:
            return f"an {self.name}"
        return f"a {self.name}"

    @property
    def plural_names(self):
        if self.name.endswith('s'):
            return f"{self.name}es"
        return f"{self.name}s"


class FOL_Property:
    def __init__(self, prop):
        self.prop = prop
    @property
    def name(self):
        if self.prop.is_negated:
            return "not " + self.prop.name
        return self.prop.name


class FOL_Entity:
    def __init__(self, name):
        self.name = name.split("::")[-1].capitalize()


class FOL:
    def __init__(self, S, V):
        self.S = S
        self.V = V

    def __str__(self):
        if isinstance(self.S, FOL_Entity):
            if isinstance(self.V, FOL_Property):
                return f"{self.S.name} is {self.V.name}"
            if isinstance(self.V, FOL_Concept):
                return f"{self.S.name} is {self.V.name_with_article}"
        if isinstance(self.S, FOL_Concept):
            if BiasedCoin.flip(1/3):
                if isinstance(self.V, FOL_Property):
                    return f"each {self.S.name} is {self.V.name}"
                if isinstance(self.V, FOL_Concept):
                    return f"each {self.S.name} is {self.V.name_with_article}"
            if BiasedCoin.flip(1/3):
                if isinstance(self.V, FOL_Property):
                    return f"every {self.S.name} is {self.V.name}"
                if isinstance(self.V, FOL_Concept):
                    return f"every {self.S.name} is {self.V.name_with_article}"
            if isinstance(self.V, FOL_Property):
                return f"{self.S.plural_names} are {self.V.name}"
            if isinstance(self.V, FOL_Concept):
                return f"{self.S.plural_names} are {self.V.plural_names}"

    def to_fol(self):
        """Return FOL string representation instead of natural language."""
        if isinstance(self.S, FOL_Entity):
            if isinstance(self.V, FOL_Property):
                # Entity + Property: rainy(Amy) or ¬slow(Amy)
                if self.V.prop.is_negated:
                    return f"¬{self.V.prop.name}({self.S.name})"
                return f"{self.V.prop.name}({self.S.name})"
            if isinstance(self.V, FOL_Concept):
                # Entity + Concept: dalpist(Amy)
                return f"{self.V.name}({self.S.name})"
        if isinstance(self.S, FOL_Concept):
            if isinstance(self.V, FOL_Property):
                # Concept + Property: ∀x(dalpist(x) → rainy(x)) or ∀x(dalpist(x) → ¬slow(x))
                if self.V.prop.is_negated:
                    return f"∀x({self.S.name}(x) → ¬{self.V.prop.name}(x))"
                return f"∀x({self.S.name}(x) → {self.V.prop.name}(x))"
            if isinstance(self.V, FOL_Concept):
                # Concept + Concept: ∀x(cat(x) → mammal(x))
                return f"∀x({self.S.name}(x) → {self.V.name}(x))"
        return str(self)  # Fallback to NL
