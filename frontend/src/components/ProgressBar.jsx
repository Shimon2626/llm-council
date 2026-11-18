// Real progress: seats finished out of seats asked, as reported by the backend.
// Stage 3 is a single call, so it shows a counter-less note instead of a fake percentage.
export default function ProgressBar({ done, total, unit }) {
  if (!total) return null;
  if (total === 1 && !done) return <span className="progress-note">{unit}…</span>;
  const pct = Math.min(100, Math.round((done / total) * 100));
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
      <span className="progress-label">{done}/{total} {unit}</span>
    </div>
  );
}
