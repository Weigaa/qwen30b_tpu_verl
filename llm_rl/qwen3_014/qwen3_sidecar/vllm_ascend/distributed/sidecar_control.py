"""Keep the fixed DP/EP serving world alive after local requests finish."""


def has_stream_work(engine, local_requests, distributed):
    if distributed:
        # This API coordinates the DP world and marks locally finished ranks
        # for execute_dummy_batch. Local request-map emptiness is insufficient
        # while peers still need this rank's experts/collectives.
        return engine.has_unfinished_requests()
    return bool(local_requests)
