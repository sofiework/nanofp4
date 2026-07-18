def propose_draft(token_ids: list[int], k: int, n: int = 3) -> list[int]:
    # token_ids = 所有已有的 token（prompt + 已生成的）
    # k = 最多 draft 几个 token
    # n = 匹配窗口大小

    # 拿最后 n 个 token 当 pattern
    pattern = token_ids[-n:]

    # take k after match
    for i in range(len(token_ids) - n - 1, -1, -1):  # find rightmost
        if pattern == token_ids[i : i + n]:
            draft = token_ids[i + n : i + n + k]
            if len(draft) > 0:
                return draft

    # 没命中 → 返回空，这轮不做 speculative，正常 decode 一个 token
    return []
