import type { ReactNode } from "react";

export function Section({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="mt-16">
      <h2 className="mb-3 text-xl font-semibold tracking-tight sm:text-2xl">{title}</h2>
      {children}
    </section>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-line bg-surface-card ${className}`}>{children}</div>
  );
}

export function Button({
  onClick,
  disabled,
  loading,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className="inline-flex select-none items-center gap-2 rounded-lg bg-holdout px-4 py-2.5 font-semibold text-white transition hover:brightness-110 active:scale-[.98] disabled:cursor-default disabled:opacity-50"
    >
      {loading && <span className="spinner h-4 w-4" aria-hidden />}
      {children}
    </button>
  );
}

/** Grounding provenance line under an answer. */
export function Facts({ children }: { children: ReactNode }) {
  return <div className="mt-3 text-[13px] leading-relaxed text-ink-mute">{children}</div>;
}

/** Left-accented output panel with an entrance animation. */
export function Output({ children }: { children: ReactNode }) {
  return (
    <div className="animate-fadeUp mt-4 rounded-lg border-l-[3px] border-holdout bg-surface px-4 py-4 leading-relaxed">
      {children}
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 min-h-[1.25rem] text-sm text-ink-mute" role="status" aria-live="polite">
      {children}
    </p>
  );
}
