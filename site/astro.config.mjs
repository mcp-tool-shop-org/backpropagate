// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://mcp-tool-shop-org.github.io',
  base: '/backpropagate',
  integrations: [
    starlight({
      title: 'Backpropagate',
      description: 'Backpropagate handbook',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/mcp-tool-shop-org/backpropagate' },
      ],
      sidebar: [
        {
          label: 'Handbook',
          // Wave 6a.0 (v1.4): starlight 0.39 removed top-level `autogenerate`
          // on sidebar groups; the autogenerate config now lives inside `items`.
          // See [[astro-6-upgrade-gotchas]] doctrine (Sovereign repo 2026-05-19).
          items: [{ autogenerate: { directory: 'handbook' } }],
        },
      ],
      customCss: ['./src/styles/starlight-custom.css'],
      disable404Route: true,
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
