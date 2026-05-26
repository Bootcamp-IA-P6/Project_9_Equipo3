export type Locale = "es" | "en";

export type Dict = {
  brand: { tagline: string };
  nav: {
    home: string;
    watch: string;
    hub: string;
    settings: string;
    shorts: string;
    subscriptions: string;
    library: string;
    history: string;
    yourVideos: string;
    moderation: string;
    database: string;
  };
  topbar: {
    searchPlaceholder: string;
    signIn: string;
    toggleSidebar: string;
    toggleTheme: string;
    toggleLanguage: string;
    languageLabel: string;
    themeLightLabel: string;
    themeDarkLabel: string;
  };
  watch: {
    defaultTitle: string;
    defaultMeta: string;
    suggestedVideo: string;
    upNext: string;
    externalOnly: string;
    loadingComments: string;
    loadingRecent: string;
    commentsCount: (n: number) => string;
    toxicDetected: (n: number) => string;
    composePlaceholder: string;
    composeAriaLabel: string;
    analyzing: string;
    liveScore: string;
    toxicity: string;
    cancel: string;
    comment: string;
    fromYoutube: string;
    justNow: string;
    posted: string;
    posting: string;
    you: string;
    flagged: string;
    demoBanner: string;
    placeholderTitleBanner: string;
    couldNotLoadVideos: string;
    failedToLoadComments: string;
    watchOnYoutube: string;
    dismiss: string;
    channelFallback: string;
  };
  hub: {
    title: string;
    threshold: string;
    eventsLogged: string;
    toxicSession: string;
    safeVsToxic: string;
    safe: string;
    toxic: string;
    recentScores: string;
    recentActions: string;
    colUser: string;
    colComment: string;
    colScore: string;
    colAction: string;
    emptyHistory: string;
  };
  settings: {
    title: string;
    activeModel: string;
    productionNote: (f1: string, gap: string) => string;
    baselinesNote: (lrF1: string, bertF1: string, bertGap: string) => string;
    productionLabel: string;
    lrLabel: string;
    bertLabel: string;
    installHint: string;
    switching: string;
    failedSwitch: string;
    couldNotLoadStatus: string;
    activeModelMsg: (name: string) => string;
    thresholdTitle: string;
    thresholdNote: string;
    quickTest: string;
    analyze: string;
    analyzing: string;
    testResult: (status: string, pct: string) => string;
    analysisFailed: string;
    defaultTestText: string;
  };
  badges: { safe: string; toxic: string };
  modelBanner: { current: (name: string, f1: string, gap: string) => string };
  time: {
    justNow: string;
    minutesAgo: (n: number) => string;
    hoursAgo: (n: number) => string;
    daysAgo: (n: number) => string;
  };
};

const es: Dict = {
  brand: { tagline: "Moderación inteligente para YouTube." },
  nav: {
    home: "Inicio",
    watch: "Ver",
    hub: "Panel",
    settings: "Ajustes",
    shorts: "Shorts",
    subscriptions: "Suscripciones",
    library: "Biblioteca",
    history: "Historial",
    yourVideos: "Tus vídeos",
    moderation: "Moderación",
    database: "Base de datos",
  },
  topbar: {
    searchPlaceholder: "Buscar más vídeos...",
    signIn: "Iniciar sesión",
    toggleSidebar: "Mostrar menú",
    toggleTheme: "Cambiar tema",
    toggleLanguage: "Cambiar idioma",
    languageLabel: "ES",
    themeLightLabel: "Claro",
    themeDarkLabel: "Oscuro",
  },
  watch: {
    defaultTitle: "Ver y moderar comentarios",
    defaultMeta: "Elige un vídeo de Siguiente para cargar y puntuar sus comentarios",
    suggestedVideo: "Vídeo sugerido",
    upNext: "Siguiente",
    externalOnly: "Solo externo",
    loadingComments: "Cargando comentarios…",
    loadingRecent: "Cargando comentarios recientes…",
    commentsCount: (n) => `${n} comentarios`,
    toxicDetected: (n) => ` · ${n} tóxicos detectados`,
    composePlaceholder: "Añade un comentario…",
    composeAriaLabel: "Escribe un comentario",
    analyzing: "Analizando…",
    liveScore: "Puntuación en vivo",
    toxicity: "Toxicidad",
    cancel: "Cancelar",
    comment: "Comentar",
    fromYoutube: "desde YouTube",
    justNow: "ahora mismo",
    posted: "Publicado",
    posting: "Publicando…",
    you: "tú",
    flagged: "Marcado para revisión",
    demoBanner: "Usando comentarios demo — añade YOUTUBE_API_KEY a .env para hilos reales de YouTube.",
    placeholderTitleBanner: "Metadatos demo — añade YOUTUBE_API_KEY a .env para títulos reales.",
    couldNotLoadVideos: "No se pudieron cargar los vídeos sugeridos",
    failedToLoadComments: "No se pudieron cargar los comentarios",
    watchOnYoutube: "Ver en YouTube (embed bloqueado)",
    dismiss: "Cerrar",
    channelFallback: "YouTube",
  },
  hub: {
    title: "Panel de moderación",
    threshold: "Umbral",
    eventsLogged: "Eventos registrados",
    toxicSession: "Tóxicos (sesión)",
    safeVsToxic: "Seguros vs Tóxicos",
    safe: "Seguro",
    toxic: "Tóxico",
    recentScores: "Puntuaciones recientes (%)",
    recentActions: "Acciones recientes",
    colUser: "Usuario",
    colComment: "Comentario",
    colScore: "Puntuación",
    colAction: "Acción",
    emptyHistory: "Publica comentarios en la página Ver para poblar el historial.",
  },
  settings: {
    title: "Ajustes",
    activeModel: "Modelo activo",
    productionNote: (f1, gap) => `Por defecto: Meta-Feature Stacking (Producción) (F1 ${f1}, gap ${gap}%).`,
    baselinesNote: (lrF1, bertF1, bertGap) =>
      `Bases: LR + TF-IDF (F1 ${lrF1}) y Frozen Toxic-BERT (F1 ${bertF1}, gap ${bertGap}%).`,
    productionLabel: "Meta-Feature Stacking (Producción)",
    lrLabel: "LR + TF-IDF",
    bertLabel: "Frozen Toxic-BERT",
    installHint:
      "Producción y BERT congelado requieren uv sync --extra hf (o Docker INSTALL_HF=1). El baseline LR usa solo joblib. La primera carga del transformer puede descargar pesos (~1 min).",
    switching: "Cambiando modelo… producción puede tardar hasta un minuto en la primera carga.",
    failedSwitch: "No se pudo cambiar de modelo",
    couldNotLoadStatus: "No se pudo cargar el estado de los modelos",
    activeModelMsg: (name) => `Modelo activo: ${name}`,
    thresholdTitle: "Umbral de toxicidad",
    thresholdNote: "los comentarios con probabilidad igual o superior se marcan como Tóxicos.",
    quickTest: "Prueba rápida",
    analyze: "Analizar",
    analyzing: "Analizando…",
    testResult: (status, pct) => `${status} — ${pct}% tóxico`,
    analysisFailed: "El análisis falló",
    defaultTestText: "Eres un idiota",
  },
  badges: { safe: "Seguro", toxic: "Tóxico" },
  modelBanner: {
    current: (name, f1, gap) => `En uso: ${name} (F1: ${f1}, Gap: ${gap}%)`,
  },
  time: {
    justNow: "ahora mismo",
    minutesAgo: (n) => `hace ${n} min`,
    hoursAgo: (n) => `hace ${n} h`,
    daysAgo: (n) => `hace ${n} d`,
  },
};

const en: Dict = {
  brand: { tagline: "Intelligent moderation for YouTube." },
  nav: {
    home: "Home",
    watch: "Watch",
    hub: "Hub",
    settings: "Settings",
    shorts: "Shorts",
    subscriptions: "Subscriptions",
    library: "Library",
    history: "History",
    yourVideos: "Your videos",
    moderation: "Moderation",
    database: "Database",
  },
  topbar: {
    searchPlaceholder: "Search more videos...",
    signIn: "Sign in",
    toggleSidebar: "Toggle sidebar",
    toggleTheme: "Toggle theme",
    toggleLanguage: "Toggle language",
    languageLabel: "EN",
    themeLightLabel: "Light",
    themeDarkLabel: "Dark",
  },
  watch: {
    defaultTitle: "Watch and moderate comments",
    defaultMeta: "Choose a video from Up next to load and score its comments",
    suggestedVideo: "Suggested video",
    upNext: "Up next",
    externalOnly: "External only",
    loadingComments: "Loading comments…",
    loadingRecent: "Loading recent comments…",
    commentsCount: (n) => `${n} comments`,
    toxicDetected: (n) => ` · ${n} toxic detected`,
    composePlaceholder: "Add a comment…",
    composeAriaLabel: "Write a comment",
    analyzing: "Analyzing…",
    liveScore: "Live score",
    toxicity: "Toxicity",
    cancel: "Cancel",
    comment: "Comment",
    fromYoutube: "from YouTube",
    justNow: "just now",
    posted: "Posted",
    posting: "Posting…",
    you: "you",
    flagged: "Flagged for review",
    demoBanner: "Using demo comments — add YOUTUBE_API_KEY to .env for real YouTube threads.",
    placeholderTitleBanner: "Demo metadata — add YOUTUBE_API_KEY to .env for real titles.",
    couldNotLoadVideos: "Could not load suggested videos",
    failedToLoadComments: "Failed to load comments",
    watchOnYoutube: "Watch on YouTube (embedding blocked)",
    dismiss: "Dismiss",
    channelFallback: "YouTube",
  },
  hub: {
    title: "Moderator Hub",
    threshold: "Threshold",
    eventsLogged: "Events logged",
    toxicSession: "Toxic (session)",
    safeVsToxic: "Safe vs Toxic",
    safe: "Safe",
    toxic: "Toxic",
    recentScores: "Recent scores (%)",
    recentActions: "Recent actions",
    colUser: "User",
    colComment: "Comment",
    colScore: "Score",
    colAction: "Action",
    emptyHistory: "Post comments on the Watch page to populate history.",
  },
  settings: {
    title: "Settings",
    activeModel: "Active model",
    productionNote: (f1, gap) => `Default: Meta-Feature Stacking (Production) (F1 ${f1}, gap ${gap}%).`,
    baselinesNote: (lrF1, bertF1, bertGap) =>
      `Baselines: LR + TF-IDF (F1 ${lrF1}) and Frozen Toxic-BERT (F1 ${bertF1}, gap ${bertGap}%).`,
    productionLabel: "Meta-Feature Stacking (Production)",
    lrLabel: "LR + TF-IDF",
    bertLabel: "Frozen Toxic-BERT",
    installHint:
      "Production and frozen BERT need uv sync --extra hf (or Docker INSTALL_HF=1). LR baseline uses joblib only. First transformer load may download weights (~1 min).",
    switching: "Switching model… production may take up to a minute on first load.",
    failedSwitch: "Failed to switch model",
    couldNotLoadStatus: "Could not load model status",
    activeModelMsg: (name) => `Active model: ${name}`,
    thresholdTitle: "Toxicity threshold",
    thresholdNote: "comments at or above this probability are flagged as Toxic.",
    quickTest: "Quick test",
    analyze: "Analyze",
    analyzing: "Analyzing…",
    testResult: (status, pct) => `${status} — ${pct}% toxic`,
    analysisFailed: "Analysis failed",
    defaultTestText: "You are an idiot",
  },
  badges: { safe: "Safe", toxic: "Toxic" },
  modelBanner: {
    current: (name, f1, gap) => `Currently using: ${name} (F1: ${f1}, Gap: ${gap}%)`,
  },
  time: {
    justNow: "just now",
    minutesAgo: (n) => `${n}m ago`,
    hoursAgo: (n) => `${n}h ago`,
    daysAgo: (n) => `${n}d ago`,
  },
};

export const DICTIONARIES: Record<Locale, Dict> = { es, en };
