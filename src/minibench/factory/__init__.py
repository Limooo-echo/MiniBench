"""Factories for assembling MiniBench agents, providers, and experiments."""

__all__ = [
    "AGENT_NAMES",
    "OpenAICompatibleAgent",
    "ProviderConfig",
    "make_agent",
    "make_agent_from_config",
    "resolve_provider",
]


def __getattr__(name: str):
    if name in {"AGENT_NAMES", "make_agent", "make_agent_from_config"}:
        from minibench.factory import agents

        return getattr(agents, name)
    if name in {"OpenAICompatibleAgent", "ProviderConfig", "resolve_provider"}:
        from minibench.factory import providers

        return getattr(providers, name)
    raise AttributeError(name)
