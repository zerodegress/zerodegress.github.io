import js from '@eslint/js'
import prettier from 'eslint-config-prettier'
import typescript from 'typescript-eslint'
import { FlatCompat } from '@eslint/eslintrc'

const compat = new FlatCompat({
  recommendedConfig: js.configs.recommended,
})

/** @type {import('eslint').Linter.FlatConfig[]} */
export default [
  {
    ignores: ['.astro/*', '**/env.d.ts'],
  },
  js.configs.recommended,
  ...compat.config({
    plugins: [],
    extends: [],
    rules: {},
  }),
  ...typescript.configs.recommended,
  prettier,
]
