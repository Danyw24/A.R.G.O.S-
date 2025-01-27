const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {"start":"_app/immutable/entry/start.GT9PfjDr.js","app":"_app/immutable/entry/app.BpzJss2w.js","imports":["_app/immutable/entry/start.GT9PfjDr.js","_app/immutable/chunks/client.CW4xU2M4.js","_app/immutable/entry/app.BpzJss2w.js","_app/immutable/chunks/preload-helper.DpQnamwV.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./chunks/0-BGoJamRr.js')),
			__memo(() => import('./chunks/1-CJkSPxXH.js')),
			__memo(() => import('./chunks/2-BotjbI-q.js').then(function (n) { return n.aF; }))
		],
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/(.*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
