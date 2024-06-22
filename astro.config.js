import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import react from '@astrojs/react'

import sitemap from '@astrojs/sitemap'

// https://astro.build/config
export default defineConfig({
  site: 'https://zerodegress.github.io',
  base: '/',
  integrations: [mdx(), sitemap(), react()],
})
