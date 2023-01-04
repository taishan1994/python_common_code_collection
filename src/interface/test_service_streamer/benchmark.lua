wrk.method = "POST"
wrk.body  = '{"text":["twinkle twinkle [MASK] star.","Happy birthday to [MASK].","the answer to life, the [MASK], and everything."]}'
wrk.headers["Content-Type"] = "application/json"