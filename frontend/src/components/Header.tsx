"use client";

import { Shield, Github, BookOpen } from "lucide-react";

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-screen-2xl mx-auto px-4 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-display font-bold text-lg text-surface-100">
              PrivMap
            </h1>
          </div>
        </div>

        {/* Tagline */}
        <div className="hidden md:block text-sm text-surface-400">
          Adaptive Differentially Private Spatial Analytics
        </div>

        {/* Links */}
        <div className="flex items-center gap-2">
          <a
            href="https://arxiv.org/abs/1601.03229"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg text-surface-400 hover:text-surface-100 hover:bg-surface-800 transition-colors"
            title="Read the Paper"
          >
            <BookOpen className="w-5 h-5" />
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg text-surface-400 hover:text-surface-100 hover:bg-surface-800 transition-colors"
            title="View on GitHub"
          >
            <Github className="w-5 h-5" />
          </a>
        </div>
      </div>
    </header>
  );
}
