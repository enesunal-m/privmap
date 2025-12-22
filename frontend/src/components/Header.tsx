"use client";

import { Github, BookOpen } from "lucide-react";

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-screen-2xl mx-auto px-4 h-14 flex items-center justify-between">
        {/* Logo - Simple text-based design */}
        <div className="flex items-center gap-2">
          <span className="font-mono text-lg font-bold tracking-tight">
            <span className="text-primary-400">Priv</span>
            <span className="text-surface-100">Map</span>
          </span>
          <span className="text-surface-600 text-xs font-mono hidden sm:inline">
            v0.1
          </span>
        </div>

        {/* Tagline */}
        <div className="hidden md:block text-sm text-surface-500 font-mono">
          ε-differential privacy for spatial data
        </div>

        {/* Links */}
        <div className="flex items-center gap-1">
          <a
            href="https://arxiv.org/abs/1601.03229"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 rounded text-xs text-surface-400 hover:text-surface-100 hover:bg-surface-800 transition-colors font-mono flex items-center gap-1.5"
            title="Read the Paper"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Paper</span>
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 rounded text-xs text-surface-400 hover:text-surface-100 hover:bg-surface-800 transition-colors font-mono flex items-center gap-1.5"
            title="View on GitHub"
          >
            <Github className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Code</span>
          </a>
        </div>
      </div>
    </header>
  );
}
