"""Model implementations and registration side effects."""


def load_default_components() -> None:
    # 显式导入触发注册
    import onlyuav.models.action_interpreter  # noqa: F401
    import onlyuav.models.channel  # noqa: F401
    import onlyuav.models.computing  # noqa: F401
    import onlyuav.models.energy  # noqa: F401
    import onlyuav.models.mobility  # noqa: F401
    import onlyuav.models.observation  # noqa: F401
    import onlyuav.models.power  # noqa: F401
    import onlyuav.models.reward  # noqa: F401
    import onlyuav.models.task  # noqa: F401
