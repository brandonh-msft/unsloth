ARG NODE_IMAGE=node:22-bookworm
FROM ${NODE_IMAGE}
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=secret,id=npm_config,target=/root/.npmrc,required=true \
    npm ci --no-audit --no-fund
COPY . .
RUN npm run build
