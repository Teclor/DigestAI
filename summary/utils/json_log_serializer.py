import json

class JsonLogSerializer:
    def serialize(self, record):
        return json.dumps(
            {
                "timestamp": record["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "level": record["level"].name,
                "module": record["module"],
                "line": record["line"],
                "message": record["message"],
                "extra": record.get("extra", {}),
            },
            ensure_ascii=False,
        )
