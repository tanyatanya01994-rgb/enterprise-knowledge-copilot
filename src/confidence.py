"""
Deterministic retrieval confidence scoring.

The confidence score is calculated only from retrieval and
evidence-verification results.

The LLM does not generate the confidence value.
"""


def calculate_confidence(
    reranked_results,
    verified_results,
    verification_status="verified",
    multi_document_query=False,
):
    """
    Calculate a deterministic retrieval confidence score.

    Signals used:
    1. Number of verified supporting chunks.
    2. Position of verified chunks in the reranked results.
    3. Concentration of verified evidence near the top.

    Evidence verification is the primary signal.

    Returns:
        dict containing:
            score: confidence percentage from 0 to 100
            level: Low / Medium / High
            verified_count: number of verified chunks
            reranked_count: number of reranked chunks
    """

    reranked_count = len(reranked_results)
    verified_count = len(verified_results)
    covered_documents = sorted({
        str(result.get("chunk", {}).get("document_name"))
        for result in verified_results
        if result.get("chunk", {}).get("document_name")
    })
    coverage_sufficient = not multi_document_query or len(covered_documents) >= 2

    # A verifier outage says nothing about retrieval quality.  Do not present
    # it as a 0% confidence score, because that implies evidence was judged
    # insufficient rather than that no judgment could be made.
    if verification_status == "verifier_unavailable":
        return {
            "score": None,
            "level": "Unavailable",
            "verified_count": verified_count,
            "reranked_count": reranked_count,
            "covered_documents": covered_documents,
            "multi_document_query": multi_document_query,
            "coverage_sufficient": coverage_sufficient,
        }

    # ========================================================
    # NO RETRIEVED OR VERIFIED EVIDENCE
    # ========================================================

    if reranked_count == 0 or verified_count == 0:

        return {
            "score": 0.0,
            "level": "Low",
            "verified_count": verified_count,
            "reranked_count": reranked_count,
            "covered_documents": covered_documents,
            "multi_document_query": multi_document_query,
            "coverage_sufficient": coverage_sufficient,
        }

    # ========================================================
    # IDENTIFY VERIFIED CHUNKS
    # ========================================================

    verified_ids = {
        result["chunk"].get("chunk_id")
        for result in verified_results
    }

    # ========================================================
    # CALCULATE RANK QUALITY
    # ========================================================
    #
    # Higher-ranked verified evidence receives more credit.
    #
    # Rank 1 -> 1.00
    # Rank 2 -> 0.50
    # Rank 3 -> 0.33
    # Rank 4 -> 0.25
    # Rank 5 -> 0.20
    #
    # This is a secondary signal only.
    # ========================================================

    rank_scores = []

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        chunk_id = result["chunk"].get("chunk_id")

        if chunk_id in verified_ids:

            rank_scores.append(
                1.0 / rank
            )

    if rank_scores:

        maximum_possible_rank_score = sum(
            1.0 / rank
            for rank in range(
                1,
                reranked_count + 1,
            )
        )

        rank_quality = (
            sum(rank_scores)
            / maximum_possible_rank_score
        )

    else:

        rank_quality = 0.0

    # ========================================================
    # EVIDENCE STRENGTH
    # ========================================================
    #
    # Verified evidence is the PRIMARY confidence signal.
    #
    # 1 verified chunk  -> 0.75
    # 2 verified chunks -> 0.90
    # 3+ verified       -> 0.95
    #
    # One verified chunk is enough to support a factual answer
    # when the evidence verifier has explicitly approved it.
    # ========================================================

    if verified_count == 1:

        evidence_strength = 0.75

    elif verified_count == 2:

        evidence_strength = 0.90

    else:

        evidence_strength = 0.95

    # ========================================================
    # COMBINE EVIDENCE + RANK QUALITY
    # ========================================================
    #
    # Evidence strength = 70%
    # Ranking quality   = 30%
    #
    # This prevents a correct answer from being rejected simply
    # because its strongest evidence was ranked #2 or #3.
    # ========================================================

    confidence = (
        (evidence_strength * 0.70)
        +
        (rank_quality * 0.30)
    )

    # A comparison needs support from at least two distinct documents. The
    # score remains explainable, but cannot clear the answer gate on one-sided
    # evidence even if that evidence ranks highly.
    if multi_document_query and not coverage_sufficient:
        confidence = min(confidence, 0.44)

    score = round(
        confidence * 100,
        1,
    )

    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    if score >= 70:

        level = "High"

    elif score >= 45:

        level = "Medium"

    else:

        level = "Low"

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "score": score,
        "level": level,
        "verified_count": verified_count,
        "reranked_count": reranked_count,
        "covered_documents": covered_documents,
        "multi_document_query": multi_document_query,
        "coverage_sufficient": coverage_sufficient,
    }
