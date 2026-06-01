import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import * as d3 from "d3";
import "./styles.css";

const ROLE_COLOR = {
  Hitter: "#d8a514",
  Pitcher: "#006b54",
  Fielder: "#6e7681",
};

const TABS = [
  ["timeline", "Player Timeline"],
  ["heatmap", "Season Similarity"],
  ["network", "Overlap Network"],
  ["ribbon", "Era Ribbon"],
  ["analysis", "Analysis Checks"],
];

const TIMELINE_SORTS = [
  ["total", "Total OAK WAR"],
  ["firstYear", "First OAK year"],
  ["lastName", "Last name"],
];

function publicPath(path) {
  return `${import.meta.env.BASE_URL}${path.replace(/^\/+/, "")}`;
}

function formatScore(value, digits = 3) {
  return Number(value || 0).toFixed(digits);
}

function playerLastName(playerName) {
  const parts = String(playerName || "").trim().split(/\s+/);
  return (parts[parts.length - 1] || "").toLowerCase();
}

function totalOakWar(player) {
  return Number(player.total_bwar_positive ?? player.total_metric_score ?? player.total_core ?? 0);
}

function careerMetric(season) {
  return Number(season.career_metric_score ?? season.career_war_score ?? season.career_core_score ?? 0);
}

function positiveBwar(season) {
  return Math.max(0, Number(season.bwar_positive ?? season.bwar ?? 0));
}

function positiveCareerBwar(season) {
  return Math.max(0, Number(season.career_bwar_positive ?? season.career_bwar ?? 0));
}

function overallWarScale(value, maxValue) {
  return maxValue > 0 ? Math.min(1, Math.max(0, Number(value || 0)) / maxValue) : 0;
}

function isYes(value) {
  return String(value || "").toUpperCase() === "Y";
}

function seasonOutcome(row) {
  const wsWin = isYes(row?.WSWin);
  const playoff = wsWin || isYes(row?.LgWin) || isYes(row?.DivWin) || isYes(row?.WCWin);
  return { playoff, wsWin };
}

function uniqueJoin(values) {
  return [...new Set(values.filter(Boolean))].join(" + ");
}

function aggregateCareerOverlayRows(rows) {
  const grouped = new Map();
  rows.forEach((row) => {
    const key = `${row.playerID}-${row.yearID}`;
    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, {
        ...row,
        teamIDs: [row.teamID],
        teamNames: [row.team_name],
        _maxMetric: careerMetric(row),
      });
      return;
    }

    const rowMetric = careerMetric(row);
    existing.teamIDs.push(row.teamID);
    existing.teamNames.push(row.team_name);
    existing.career_bwar = Number(existing.career_bwar || 0) + Number(row.career_bwar || 0);
    existing.career_batting_bwar = Number(existing.career_batting_bwar || 0) + Number(row.career_batting_bwar || 0);
    existing.career_pitching_bwar = Number(existing.career_pitching_bwar || 0) + Number(row.career_pitching_bwar || 0);
    existing.career_bwar_positive = Number(existing.career_bwar_positive || 0) + Number(row.career_bwar_positive || 0);
    existing.career_war_score = Math.min(1, Number(existing.career_war_score || 0) + Number(row.career_war_score || 0));
    existing.career_metric_score = Math.min(1, Number(existing.career_metric_score || 0) + Number(row.career_metric_score || 0));
    existing.career_core_score = Math.min(1, Number(existing.career_core_score || 0) + Number(row.career_core_score || 0));
    if (rowMetric > existing._maxMetric) {
      existing.primary_role = row.primary_role;
      existing.primary_pos = row.primary_pos;
      existing.career_war_rank = row.career_war_rank;
      existing.career_metric_rank = row.career_metric_rank;
      existing.career_core_rank = row.career_core_rank;
      existing._maxMetric = rowMetric;
    }
    existing.bat_PA_est = Number(existing.bat_PA_est || 0) + Number(row.bat_PA_est || 0);
    existing.pit_IP = Number(existing.pit_IP || 0) + Number(row.pit_IP || 0);
    existing.app_G_all = Math.max(Number(existing.app_G_all || 0), Number(row.app_G_all || 0));
  });

  return [...grouped.values()].map((row) => ({
    ...Object.fromEntries(Object.entries(row).filter(([key]) => key !== "_maxMetric")),
    teamID: uniqueJoin(row.teamIDs),
    team_name: uniqueJoin(row.teamNames),
  }));
}

function sortPlayers(players, sortMode) {
  const sorted = [...players];
  if (sortMode === "firstYear") {
    return sorted.sort((a, b) =>
      d3.ascending(a.first_year, b.first_year)
      || d3.descending(totalOakWar(a), totalOakWar(b))
      || d3.ascending(playerLastName(a.player_name), playerLastName(b.player_name))
      || d3.ascending(a.player_name, b.player_name)
    );
  }
  if (sortMode === "lastName") {
    return sorted.sort((a, b) =>
      d3.ascending(playerLastName(a.player_name), playerLastName(b.player_name))
      || d3.ascending(a.player_name, b.player_name)
      || d3.ascending(a.first_year, b.first_year)
    );
  }
  return sorted.sort((a, b) =>
    d3.ascending(a.overall_rank, b.overall_rank)
    || d3.descending(totalOakWar(a), totalOakWar(b))
  );
}

function useAppData() {
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  useEffect(() => {
    fetch(`${publicPath("data/app_data.json")}?v=${Date.now()}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => setState({ status: "ready", data, error: null }))
      .catch((error) => setState({ status: "error", data: null, error }));
  }, []);
  return state;
}

function Tooltip({ tooltip }) {
  if (!tooltip) return null;
  return (
    <div className="tooltip" style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}>
      <strong>{tooltip.title}</strong>
      {tooltip.lines.map((line) => (
        <span key={line}>{line}</span>
      ))}
    </div>
  );
}

function Header({ data }) {
  const seasons = data.meta.year_end - data.meta.year_start + 1;
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">Lahman rosters plus Baseball-Reference bWAR, 1968-2024</p>
        <h1>Oakland A's Roster Eras</h1>
      </div>
      <div className="stat-strip" aria-label="Dataset summary">
        <span><b>{seasons}</b> seasons</span>
        <span><b>{data.playerSeasons.length.toLocaleString()}</b> player-seasons</span>
        {data.careerSeasons && <span><b>{data.careerSeasons.length.toLocaleString()}</b> MLB team-seasons</span>}
        <span><b>{data.playerTotals.length.toLocaleString()}</b> ranked players</span>
        <a href={data.meta.source_page} target="_blank" rel="noreferrer">SABR Lahman</a>
        <a href={data.meta.bref_batting_war} target="_blank" rel="noreferrer">BRef bWAR</a>
      </div>
    </header>
  );
}

function Controls({ data, topN, setTopN, era, setEra, query, setQuery }) {
  const eras = data.analysis.candidate_eras;
  return (
    <section className="controls" aria-label="Visualization controls">
      <label>
        <span>Top players</span>
        <input
          type="range"
          min="20"
          max="180"
          step="10"
          value={topN}
          onChange={(event) => setTopN(Number(event.target.value))}
        />
        <b>{topN}</b>
      </label>
      <label>
        <span>Candidate era</span>
        <select value={era} onChange={(event) => setEra(event.target.value)}>
          <option value="all">All eras</option>
          {eras.map((item) => (
            <option key={item.era_id} value={item.era_id}>{item.label}</option>
          ))}
        </select>
      </label>
      <label className="search-box">
        <span>Player search</span>
        <input
          type="search"
          value={query}
          placeholder="Giambi, Henderson, Zito..."
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="download-links">
        <a href={publicPath("exports/player_timeline_top60.svg")}>Timeline SVG</a>
        <a href={publicPath("exports/player_timeline_career_top60.svg")}>Career SVG</a>
        <a href={publicPath("exports/season_similarity_heatmap.svg")}>Heatmap SVG</a>
        <a href={publicPath("exports/player_overlap_network.svg")}>Network SVG</a>
        <a href={publicPath("exports/era_ribbon_top12.svg")}>Ribbon SVG</a>
      </div>
    </section>
  );
}

function Tabs({ active, setActive }) {
  return (
    <nav className="tabs" aria-label="Views">
      {TABS.map(([id, label]) => (
        <button
          key={id}
          className={active === id ? "active" : ""}
          type="button"
          onClick={() => setActive(id)}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}

function PlayerTimeline({ data, players, query, timelineSort, setTimelineSort, showCareer, setShowCareer }) {
  const [tooltip, setTooltip] = useState(null);
  const years = d3.range(data.meta.year_start, data.meta.year_end + 1);
  const playerIds = new Set(players.map((player) => player.playerID));
  const rows = useMemo(
    () => data.playerSeasons.filter((row) => playerIds.has(row.playerID)),
    [data.playerSeasons, players],
  );
  const visualWarMax = useMemo(() => {
    const oakMax = d3.max(data.playerSeasons, positiveBwar) || 0;
    const careerMax = d3.max(
      aggregateCareerOverlayRows((data.careerSeasons || []).filter((row) => !row.is_oakland)),
      positiveCareerBwar,
    ) || 0;
    return Math.max(oakMax, careerMax, 1);
  }, [data.playerSeasons, data.careerSeasons]);
  const careerRows = useMemo(
    () => showCareer
      ? aggregateCareerOverlayRows((data.careerSeasons || []).filter((row) => playerIds.has(row.playerID) && !row.is_oakland))
      : [],
    [data.careerSeasons, players, playerIds, showCareer],
  );
  const byPlayer = useMemo(() => d3.group(rows, (row) => row.playerID), [rows]);
  const careerByPlayer = useMemo(() => d3.group(careerRows, (row) => row.playerID), [careerRows]);
  const seasonOutcomes = useMemo(
    () => new Map(data.seasonSummary.map((row) => [row.yearID, seasonOutcome(row)])),
    [data.seasonSummary],
  );
  const width = 1180;
  const left = 180;
  const right = 20;
  const top = 36;
  const axisHeight = 42;
  const rowHeight = 20;
  const height = top + players.length * rowHeight + 34;
  const x = d3.scaleLinear()
    .domain([data.meta.year_start - 0.5, data.meta.year_end + 0.5])
    .range([left, width - right]);
  const queryLower = query.trim().toLowerCase();

  return (
    <section className="viz-panel timeline-panel">
      <div className="panel-heading">
        <div>
          <h2>Player Timeline</h2>
          <p>Oakland bars use an overall bWAR scale; optional gray bars show non-Oakland career seasons on the same scale.</p>
        </div>
        <div className="timeline-tools">
          <label>
            <span>Sort rows</span>
            <select value={timelineSort} onChange={(event) => setTimelineSort(event.target.value)}>
              {TIMELINE_SORTS.map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={showCareer}
              onChange={(event) => setShowCareer(event.target.checked)}
            />
            <span>Show full MLB career</span>
          </label>
        </div>
      </div>
      <div className="timeline-page-wrap">
        <div className="timeline-sticky-axis">
          <svg viewBox={`0 0 ${width} ${axisHeight}`} role="presentation" aria-hidden="true">
            {years.map((year) => {
              const outcome = seasonOutcomes.get(year) || {};
              if (!outcome.playoff) return null;
              return (
                <rect
                  key={`axis-shade-${year}`}
                  className={outcome.wsWin ? "season-shade ws-win" : "season-shade playoff"}
                  x={x(year - 0.5)}
                  y="0"
                  width={x(year + 0.5) - x(year - 0.5)}
                  height={axisHeight}
                />
              );
            })}
            <line className="axis-baseline" x1={left} x2={width - right} y1={axisHeight - 1} y2={axisHeight - 1} />
            {years.filter((year) => year % 5 === 0).map((year) => (
              <g className="timeline-year-tick" key={`axis-${year}`}>
                <line x1={x(year)} x2={x(year)} y1="0" y2={axisHeight - 1} />
                <text x={x(year)} y="18">{year}</text>
              </g>
            ))}
          </svg>
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Oakland player timeline">
          <g className="season-shades">
            {years.map((year) => {
              const outcome = seasonOutcomes.get(year) || {};
              if (!outcome.playoff) return null;
              return (
                <rect
                  key={`shade-${year}`}
                  className={outcome.wsWin ? "season-shade ws-win" : "season-shade playoff"}
                  x={x(year - 0.5)}
                  y="0"
                  width={x(year + 0.5) - x(year - 0.5)}
                  height={height}
                />
              );
            })}
          </g>
          <g className="grid">
            {years.filter((year) => year % 5 === 0).map((year) => (
              <g key={year}>
                <line x1={x(year)} x2={x(year)} y1="20" y2={height - 26} />
              </g>
            ))}
          </g>
          {players.map((player, index) => {
            const y = top + index * rowHeight;
            const highlighted = queryLower && player.player_name.toLowerCase().includes(queryLower);
            return (
              <g key={player.playerID}>
                <text className={highlighted ? "row-label highlight" : "row-label"} x="8" y={y + 5}>
                  {player.player_name}
                </text>
                <line className="row-rule" x1={left} x2={width - right} y1={y} y2={y} />
                {(careerByPlayer.get(player.playerID) || []).map((season) => {
                  const score = overallWarScale(positiveCareerBwar(season), visualWarMax);
                  const barHeight = 4 + score * 11;
                  return (
                    <rect
                      className="career-bar"
                      key={`${player.playerID}-${season.yearID}-${season.teamID}`}
                      x={x(season.yearID - 0.42)}
                      y={y - barHeight / 2}
                      width={Math.max(2, x(season.yearID + 0.42) - x(season.yearID - 0.42))}
                      height={barHeight}
                      rx="1"
                      fill="#7f8580"
                      opacity={0.22 + score * 0.68}
                      onMouseMove={(event) => setTooltip({
                        x: event.clientX,
                        y: event.clientY,
                        title: `${season.player_name}, ${season.yearID}`,
                        lines: [
                          `${season.team_name} (${season.teamID})`,
                          `bWAR ${formatScore(season.career_bwar, 1)}; overall scale ${formatScore(score)}`,
                          `Team WAR rank ${season.career_war_rank}; team score ${formatScore(season.career_war_score)}`,
                          `${season.primary_role}${season.primary_pos ? ` (${season.primary_pos})` : ""}`,
                          `Playing-time core ${formatScore(season.career_core_score)}`,
                          `PA ${Math.round(season.bat_PA_est || 0)}, IP ${formatScore(season.pit_IP || 0, 1)}`,
                        ],
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    />
                  );
                })}
                {(byPlayer.get(player.playerID) || []).map((season) => {
                  const score = overallWarScale(positiveBwar(season), visualWarMax);
                  const barHeight = 4 + score * 11;
                  return (
                    <rect
                      key={`${player.playerID}-${season.yearID}`}
                      x={x(season.yearID - 0.45)}
                      y={y - barHeight / 2}
                      width={Math.max(3, x(season.yearID + 0.45) - x(season.yearID - 0.45))}
                      height={barHeight}
                      rx="1"
                      fill={ROLE_COLOR[season.primary_role] || ROLE_COLOR.Fielder}
                      opacity={0.28 + score * 0.72}
                      onMouseMove={(event) => setTooltip({
                        x: event.clientX,
                        y: event.clientY,
                        title: `${season.player_name}, ${season.yearID}`,
                        lines: [
                          `bWAR ${formatScore(season.bwar, 1)}; overall scale ${formatScore(score)}`,
                          `Season WAR rank ${season.war_rank}; season score ${formatScore(season.war_score)}`,
                          `${season.primary_role}${season.primary_pos ? ` (${season.primary_pos})` : ""}`,
                          `Playing-time core ${formatScore(season.core_score)}`,
                          `PA ${Math.round(season.bat_PA_est || 0)}, IP ${formatScore(season.pit_IP || 0, 1)}`,
                        ],
                      })}
                      onMouseLeave={() => setTooltip(null)}
                    />
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <Tooltip tooltip={tooltip} />
    </section>
  );
}

function SimilarityHeatmap({ data }) {
  const [tooltip, setTooltip] = useState(null);
  const years = d3.range(data.meta.year_start, data.meta.year_end + 1);
  const simMap = useMemo(() => {
    const map = new Map();
    data.similarity.forEach((row) => map.set(`${row.year_a}-${row.year_b}`, row.similarity));
    return map;
  }, [data.similarity]);
  const summaryByYear = useMemo(() => new Map(data.seasonSummary.map((row) => [row.yearID, row])), [data.seasonSummary]);
  const size = 780;
  const pad = 70;
  const cell = (size - pad - 20) / years.length;
  const color = d3.scaleLinear()
    .domain([0, 0.18, 0.42, 0.65])
    .range(["#f4f5ee", "#d9e2c9", "#74a56d", "#003831"]);

  return (
    <section className="viz-panel heatmap-layout">
      <div>
        <div className="panel-heading">
          <h2>Season Similarity Heatmap</h2>
          <p>Weighted Jaccard similarity compares positive bWAR core vectors across seasons.</p>
        </div>
        <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Season similarity heatmap">
          {years.map((year, i) => year % 5 === 0 && (
            <g key={`tick-${year}`}>
              <text className="axis-label" x={pad + i * cell + cell / 2} y="52" textAnchor="middle" transform={`rotate(-65 ${pad + i * cell + cell / 2} 52)`}>{year}</text>
              <text className="axis-label" x="58" y={pad + i * cell + cell / 2 + 3} textAnchor="end">{year}</text>
            </g>
          ))}
          {years.map((yearA, i) => years.map((yearB, j) => {
            const value = simMap.get(`${yearA}-${yearB}`) || 0;
            return (
              <rect
                key={`${yearA}-${yearB}`}
                x={pad + j * cell}
                y={pad + i * cell}
                width={cell + 0.2}
                height={cell + 0.2}
                fill={color(value)}
                onMouseMove={(event) => {
                  const a = summaryByYear.get(yearA);
                  const b = summaryByYear.get(yearB);
                  setTooltip({
                    x: event.clientX,
                    y: event.clientY,
                    title: `${yearA} vs ${yearB}`,
                    lines: [
                      `Similarity ${formatScore(value)}`,
                      `${yearA}: ${a?.top10_players?.split(" | ").slice(0, 3).join(", ") || ""}`,
                      `${yearB}: ${b?.top10_players?.split(" | ").slice(0, 3).join(", ") || ""}`,
                    ],
                  });
                }}
                onMouseLeave={() => setTooltip(null)}
              />
            );
          }))}
        </svg>
      </div>
      <aside className="side-list">
        <h3>Largest Adjacent Turnover</h3>
        {data.analysis.biggest_adjacent_turnover.slice(0, 10).map((row) => (
          <div className="rank-row" key={`${row.year}-${row.next_year}`}>
            <span>{row.year}-{row.next_year}</span>
            <b>{formatScore(row.turnover)}</b>
          </div>
        ))}
        <h3>Closest Season Pairs</h3>
        {data.analysis.most_similar_seasons.slice(0, 10).map((row) => (
          <div className="rank-row" key={`${row.year_a}-${row.year_b}`}>
            <span>{row.year_a} & {row.year_b}</span>
            <b>{formatScore(row.similarity)}</b>
          </div>
        ))}
      </aside>
      <Tooltip tooltip={tooltip} />
    </section>
  );
}

function OverlapNetwork({ data, query }) {
  const [tooltip, setTooltip] = useState(null);
  const [minWeight, setMinWeight] = useState(0.45);
  const nodes = data.networkNodes;
  const nodeMap = useMemo(() => new Map(nodes.map((node) => [node.playerID, node])), [nodes]);
  const edges = data.networkEdges.filter((edge) => edge.weighted_core >= minWeight && nodeMap.has(edge.source) && nodeMap.has(edge.target));
  const width = 1080;
  const height = 680;
  const x = (node) => 70 + node.layout_x * (width - 140);
  const y = (node) => 55 + node.layout_y * (height - 110);
  const radius = d3.scaleSqrt()
    .domain(d3.extent(nodes, (node) => totalOakWar(node)))
    .range([4, 18]);
  const color = d3.scaleOrdinal(d3.schemeTableau10);
  const queryLower = query.trim().toLowerCase();

  return (
    <section className="viz-panel">
      <div className="panel-heading network-heading">
        <div>
          <h2>Player Overlap Network</h2>
          <p>Edges connect players who shared Oakland seasons; heavier edges reflect stronger shared bWAR core years.</p>
        </div>
        <label className="compact-range">
          <span>Min edge</span>
          <input min="0" max="3" step="0.05" type="range" value={minWeight} onChange={(event) => setMinWeight(Number(event.target.value))} />
          <b>{formatScore(minWeight, 2)}</b>
        </label>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Player overlap network">
        <g className="network-axis">
          {d3.range(1970, 2025, 10).map((year) => {
            const px = 70 + ((year - data.meta.year_start) / (data.meta.year_end - data.meta.year_start)) * (width - 140);
            return (
              <g key={year}>
                <line x1={px} x2={px} y1="34" y2={height - 26} />
                <text x={px} y={height - 6}>{year}</text>
              </g>
            );
          })}
        </g>
        <g>
          {edges.map((edge) => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            return (
              <line
                key={`${edge.source}-${edge.target}`}
                x1={x(source)}
                y1={y(source)}
                x2={x(target)}
                y2={y(target)}
                stroke="#56645d"
                strokeWidth={0.4 + Math.min(edge.weighted_core, 4) * 0.55}
                opacity={0.14 + Math.min(edge.weighted_core, 4) * 0.09}
              />
            );
          })}
        </g>
        <g>
          {nodes.map((node) => {
            const highlighted = queryLower && node.player_name.toLowerCase().includes(queryLower);
            return (
              <g key={node.playerID}>
                <circle
                  cx={x(node)}
                  cy={y(node)}
                  r={radius(totalOakWar(node)) + (highlighted ? 5 : 0)}
                  fill={color(node.community_id)}
                  stroke={highlighted ? "#111" : "#173b34"}
                  strokeWidth={highlighted ? 2.4 : 0.8}
                  opacity={0.92}
                  onMouseMove={(event) => setTooltip({
                    x: event.clientX,
                    y: event.clientY,
                    title: node.player_name,
                    lines: [
                      `Community ${node.community_id}: ${node.community_label}`,
                      `Oakland ${node.first_year}-${node.last_year}; peak ${node.peak_year}`,
                      `Total OAK bWAR ${formatScore(node.total_bwar, 1)}`,
                      `Playing-time core ${formatScore(node.total_core, 2)}`,
                    ],
                  })}
                  onMouseLeave={() => setTooltip(null)}
                />
                {node.overall_rank <= 42 && (
                  <text className="node-label" x={x(node) + radius(totalOakWar(node)) + 4} y={y(node) + 3}>
                    {node.player_name}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      <Tooltip tooltip={tooltip} />
    </section>
  );
}

function EraRibbon({ data }) {
  const [tooltip, setTooltip] = useState(null);
  const years = d3.range(data.meta.year_start, data.meta.year_end + 1);
  const width = 1200;
  const height = 520;
  const pad = { top: 36, right: 28, bottom: 38, left: 54 };
  const x = d3.scaleLinear().domain([data.meta.year_start, data.meta.year_end]).range([pad.left, width - pad.right]);
  const y = d3.scaleLinear().domain([1, 12]).range([pad.top, height - pad.bottom]);
  const byPlayer = d3.group(data.ribbon, (row) => row.playerID);
  const color = d3.scaleOrdinal(d3.schemeTableau10);
  const line = d3.line()
    .x((row) => x(row.yearID))
    .y((row) => y(row.metric_rank ?? row.core_rank))
    .curve(d3.curveMonotoneX);

  return (
    <section className="viz-panel">
      <div className="panel-heading">
        <h2>Era Ribbon Chart</h2>
        <p>Each year shows the top 12 bWAR core players; continuous lines reveal where cores genuinely persisted.</p>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Era ribbon chart">
        <g className="grid">
          {years.filter((year) => year % 5 === 0).map((year) => (
            <g key={year}>
              <line x1={x(year)} x2={x(year)} y1={pad.top - 12} y2={height - pad.bottom + 10} />
              <text x={x(year)} y={height - 10}>{year}</text>
            </g>
          ))}
          {d3.range(1, 13).map((rank) => (
            <g key={rank}>
              <line x1={pad.left} x2={width - pad.right} y1={y(rank)} y2={y(rank)} opacity="0.42" />
              <text x={24} y={y(rank) + 4}>#{rank}</text>
            </g>
          ))}
        </g>
        {[...byPlayer.entries()].map(([playerID, rows]) => {
          const sorted = rows.sort((a, b) => a.yearID - b.yearID);
          const segments = [];
          let current = [];
          sorted.forEach((row, index) => {
            if (index === 0 || row.yearID === sorted[index - 1].yearID + 1) {
              current.push(row);
            } else {
              if (current.length > 1) segments.push(current);
              current = [row];
            }
          });
          if (current.length > 1) segments.push(current);
          return segments.map((segment, index) => (
            <path
              key={`${playerID}-${index}`}
              d={line(segment)}
              fill="none"
              stroke={color(playerID)}
              strokeWidth={2 + d3.mean(segment, (row) => row.metric_score ?? row.core_score) * 5}
              opacity="0.58"
              strokeLinecap="round"
              onMouseMove={(event) => setTooltip({
                x: event.clientX,
                y: event.clientY,
                title: segment[0].player_name,
                lines: [
                  `${segment[0].yearID}-${segment[segment.length - 1].yearID}`,
                  `WAR ranks ${segment.map((row) => `#${row.metric_rank ?? row.core_rank}`).join(", ")}`,
                ],
              })}
              onMouseLeave={() => setTooltip(null)}
            />
          ));
        })}
        {data.ribbon.filter((row) => [1968, 1972, 1989, 2002, 2012, 2024].includes(row.yearID) && (row.metric_rank ?? row.core_rank) <= 8).map((row) => (
          <text className="ribbon-label" key={`${row.playerID}-${row.yearID}`} x={x(row.yearID) + 4} y={y(row.metric_rank ?? row.core_rank) + 3}>
            {row.player_name}
          </text>
        ))}
      </svg>
      <Tooltip tooltip={tooltip} />
    </section>
  );
}

function AnalysisChecks({ data }) {
  const analysis = data.analysis;
  return (
    <section className="viz-panel analysis-grid">
      <article>
        <h2>Specific Example Checks</h2>
        {analysis.specific_checks.map((check) => (
          <div className="check-block" key={check.pair}>
            <h3>{check.pair}</h3>
            <p>
              {check.overlapped
                ? `Overlapped in ${check.overlap_years.join(", ")} with weighted overlap ${formatScore(check.weighted_overlap)}.`
                : "No Oakland roster overlap in the Lahman player-season records."}
            </p>
            {check.details?.length > 0 && (
              <table>
                <thead><tr><th>Year</th><th>A bWAR</th><th>B bWAR</th><th>A score</th><th>B score</th><th>A PA</th><th>B PA</th></tr></thead>
                <tbody>
                  {check.details.map((row) => (
                    <tr key={row.yearID}>
                      <td>{row.yearID}</td>
                      <td>{formatScore(row.a_bwar, 1)}</td>
                      <td>{formatScore(row.b_bwar, 1)}</td>
                      <td>{formatScore(row.a_score)}</td>
                      <td>{formatScore(row.b_score)}</td>
                      <td>{Math.round(row.a_pa)}</td>
                      <td>{Math.round(row.b_pa)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </article>
      <article>
        <h2>Bridge Players</h2>
        <p className="small-note">Window check: 1988-1997 against 2000-2004, ranked by the weaker side of the bridge.</p>
        {analysis.bridge_players_late_mcgwire_to_early_moneyball.map((row) => (
          <div className="rank-row" key={row.playerID}>
            <span>{row.player_name}</span>
            <b>{formatScore(row.bridge_score)}</b>
          </div>
        ))}
      </article>
      <article>
        <h2>Same Cluster, No Overlap</h2>
        {analysis.same_cluster_no_overlap.slice(0, 10).map((row) => (
          <div className="compact-pair" key={`${row.player_a}-${row.player_b}`}>
            <b>{row.player_a}</b> / <b>{row.player_b}</b>
            <span>Peaks {row.peak_years}; common teammate score {formatScore(row.common_teammate_score)}</span>
          </div>
        ))}
      </article>
      <article>
        <h2>Technical Overlap, Different Peaks</h2>
        {analysis.technical_overlap_different_peak_eras.slice(0, 10).map((row) => (
          <div className="compact-pair" key={`${row.player_a}-${row.player_b}`}>
            <b>{row.player_a}</b> / <b>{row.player_b}</b>
            <span>{row.shared_seasons} shared season(s), peak gap {row.peak_year_gap} years</span>
          </div>
        ))}
      </article>
    </section>
  );
}

function App() {
  const { status, data, error } = useAppData();
  const initialTab = () => {
    const hash = window.location.hash.replace("#", "");
    return TABS.some(([id]) => id === hash) ? hash : "timeline";
  };
  const [active, setActiveState] = useState(initialTab);
  const setActive = (id) => {
    window.location.hash = id;
    setActiveState(id);
  };
  const [topN, setTopN] = useState(80);
  const [era, setEra] = useState("all");
  const [query, setQuery] = useState("");
  const [timelineSort, setTimelineSort] = useState("total");
  const [showCareer, setShowCareer] = useState(false);

  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.replace("#", "");
      if (TABS.some(([id]) => id === hash)) setActiveState(hash);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const players = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    const filtered = data.playerTotals.filter((player) => {
      const eraOk = era === "all" || player.peak_era_id === era;
      const queryOk = !q || player.player_name.toLowerCase().includes(q);
      return eraOk && queryOk;
    });
    return sortPlayers(filtered.slice(0, topN), timelineSort);
  }, [data, era, query, timelineSort, topN]);

  if (status === "loading") return <main className="loading">Loading Oakland roster eras...</main>;
  if (status === "error") return <main className="loading">Could not load data: {error.message}</main>;

  return (
    <main>
      <Header data={data} />
      <Controls data={data} topN={topN} setTopN={setTopN} era={era} setEra={setEra} query={query} setQuery={setQuery} />
      <Tabs active={active} setActive={setActive} />
      {active === "timeline" && (
        <PlayerTimeline
          data={data}
          players={players}
          query={query}
          timelineSort={timelineSort}
          setTimelineSort={setTimelineSort}
          showCareer={showCareer}
          setShowCareer={setShowCareer}
        />
      )}
      {active === "heatmap" && <SimilarityHeatmap data={data} />}
      {active === "network" && <OverlapNetwork data={data} query={query} />}
      {active === "ribbon" && <EraRibbon data={data} />}
      {active === "analysis" && <AnalysisChecks data={data} />}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
