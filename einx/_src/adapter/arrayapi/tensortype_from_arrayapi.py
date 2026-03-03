import einx._src.tracer as tracer


def tensortype_from_arrayapi(xp):
    if hasattr(xp, "_get_xp"):
        typing = tracer.signature.python.import_("typing")
        type = typing.get_type_hints(xp._get_xp().asarray)["return"]
    else:
        import typing

        type = typing.get_type_hints(xp.asarray)["return"]
    return type
