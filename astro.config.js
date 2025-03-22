import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import solidjs from '@astrojs/solid-js'
import tailwindcss from '@tailwindcss/vite'

import sitemap from '@astrojs/sitemap'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

// https://astro.build/config
export default defineConfig({
  site: 'https://www.zerodegress.ink',
  integrations: [mdx(), sitemap(), solidjs()],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    plugins: [tailwindcss()],
  },
})
