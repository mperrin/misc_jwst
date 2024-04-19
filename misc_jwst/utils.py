def get_visitid(visitstr):
    """ Common util function to handle several various kinds of visit specification"""
    if visitstr.startswith("V"):
        return visitstr
    elif ':' in visitstr:
        # This is PPS format visit ID, like 4503:31:1
        parts = [int(p) for p in visitstr.split(':')]
        return f"V{parts[0]:05d}{parts[1]:03d}{parts[2]:03d}"

