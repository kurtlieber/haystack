"""haystack genres prints the liked songs' styles (kurtlieber/dotfiles#350: the config goes before the top count)."""
import json

import numpy as np

from haystack_music import cli, config, db, embed, vectors


def _world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMB_DIR", tmp_path / "emb")
    path = tmp_path / "t.db"
    con = db.connect(path)
    for i, style_idx in enumerate((7, 7, 3), start=1):
        sid = f"liked{i}".ljust(22, "0")
        con.execute("INSERT INTO likes(spotify_id, added_at) VALUES(?, '2026-01-01')", (sid,))
        con.execute("INSERT INTO liked_map(spotify_id, deezer_id, vector_status) VALUES(?, ?, 'ok')", (sid, i))
        style = np.zeros(400)
        style[style_idx] = 1.0
        vectors.put_deezer_vector(con, i, np.ones(512), np.zeros(1280), 10, 30.0, style)
    con.commit()
    con.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(embed, "style_classes", lambda: ["Parent---Style %d" % i for i in range(400)])
    return config.load(tmp_path / "none.toml")


def test_cli_genres_prints_the_styles(tmp_path, monkeypatch, capsys):
    cfg = _world(tmp_path, monkeypatch)
    p = cli.build_parser()
    assert cli.cmd_genres(cfg, p.parse_args(["genres", "--top", "2"])) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["liked_with_styles"] == 3
    assert out["top1_style_counts"] == [["Parent / Style 7", 2], ["Parent / Style 3", 1]]
    assert len(out["top_styles_mean_activation"]) == 2 and out["top_styles_mean_activation"][0][0] == "Parent / Style 7"
    assert out["parent_genre_share"] == [["Parent", 1.0]]
    assert cli.cmd_genres(cfg, p.parse_args(["genres"])) == 0          # the default top of 15
    assert len(json.loads(capsys.readouterr().out)["top_styles_mean_activation"]) == 15
