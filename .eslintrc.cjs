/* eslint-env node */
module.exports = {
    extends: [
        'eslint:recommended', 
        'plugin:@typescript-eslint/recommended',
        'prettier',
        'plugin:solid/typescript'
    ],
    parser: '@typescript-eslint/parser',
    plugins: ['@typescript-eslint', 'solid'],
    root: true,
};