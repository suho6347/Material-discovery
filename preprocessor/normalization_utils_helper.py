
title_keyword = ['Foreword', 'Prelude', 'Commentary',
                'Workshop', 'Conference', 'Symposium',
                'Comment', 'Retract', 'Correction',
                'Erratum', 'Memorial', "Withdrawn"]

abstract_keyword = ["<\n", "Abstract\n", "Abstracts\n", "Project\n", "Abstracts",
                "Procedure\n", "Experimental\n", "Hypothesis\n", "Background\n", 
                "Backgrounds\n", "Backgrounds:", "Backgrounds\.", "Backgroud\n", 
                "Backround\n", "Backgrorund\n", "Background:", "Background\.", 
                "Background/aims\n", "Background/aim\n", "Background and aim\n", 
                "Background and aims\n", "Background and object\n", "Background and objective\n", 
                "Background and objectives\n", "Background and purpose\n",
                "Background and Purpose\n", "Findings\n", "Finding\n",
                "Findings:", "Finding:", "Findings\.", "Finding\.", "Design\n", 
                "Design\.", "Design and methods\n", "Design and Methods\n", 
                "Design &amp; methods\n", "Study design\n", "Methods\n", 
                "Methods:", "Methods\.", "Method\n", "Method:", "Method\.", 
                "Methods and results\n", "Results\n", "Results:", "Results\.",
                "Result\n", "Result:", "Result\.", "Interpretations\n",
                "Interpretation\n", "Interpretations:", "Interpretation:",
                "Interpretations\.", "Interpretation\.", "Conclusions\n",
                "Conclusions:", "Conclusions\.", "Conclusion\n",
                "Conclusion:", "Conclusion\.", "Objective\n", 
                "Objectives\n", "Objective\.", "Objectives\.", "Introduction\n", 
                "Keywords\n", "Materials and methods\n", "Materials and Methods\n",
                "Material and methods\n", "Material and Methods\n", "Case report\n",
                "Patients and methods\n", "Methods and results\n", "Methods and results:", 
                "Background and objectives\n", "Aim\n", "Aim:", "Aims\n", 
                "Aims:", "Aims and objective\n", "Aim and methods\n",
                "Aim of the study\n", "Subjects and methods\n", 
                "Results and conclusion\n", "Results and conclusions\n", 
                "Patients/methods\n", "Discussion\n", "Results and discussion\n", 
                "Context and objective\n", "Context\n", "Purpose\n", "Methodology\n",
                "Rationale\n", "Résumé\n", "Advances in knowledge\n", 
                "Relevance\.", "Relevance\n", "Relevance\ ", 
                "In this paper, ", "In this work, ", "In this study, "]

html_mappings = {
            # https://www.webatic.com/html-entities-table
                # HTML Entities Table - Latin characters
                "&nbsp;"		:' ',
                "&iexcl;"		:'¡',
                "&cent;"		:'¢',
                "&pound;"		:'£',
                "&curren;"		:'¤',
                "&yen;"			:'¥',
                "&brvbar;"		:'¦',
                "&sect;"		:'§',
                "&uml;"			:'¨',
                "&copy;"		:'©',
                "&ordf;"		:'ª',
                "&laquo;"		:'«',
                "&not;"			:'¬',
                "&shy;"			:'­',
                "&reg;"			:'®',
                "&macr;"		:'¯',
                "&deg;"			:'°',
                "&plusmn;"		:'±',
                "&sup2;"		:'²',
                "&sup3;"		:'³',
                "&acute;"		:'´',
                "&micro;"		:'µ',
                "&para;"		:'¶',
                "&middot;"		:'·',
                "&cedil;"		:'¸',
                "&sup1;"		:'¹',
                "&ordm;"		:'º',
                "&raquo;"		:'»',
                "&frac14;"		:'¼',
                "&frac12;"		:'½',
                "&frac34;"		:'¾',
                "&iquest;"		:'¿',
                "&Agr;"			:'À',
                "&Agrave;"		:'À',
                "&Aacute;"		:'Á',
                "&Acirc;"		:'Â',
                "&Atilde;"		:'Ã',
                "&Auml;"		:'Ä',
                "&Aring;"		:'Å',
                "&AElig;"		:'Æ',
                "&Ccedil;"		:'Ç',
                "&Egrave;"		:'È',
                "&Eacute;"		:'É',
                "&Ecirc;"		:'Ê',
                "&Euml;"		:'Ë',
                "&Igrave;"		:'Ì',
                "&Iacute;"		:'Í',
                "&Icirc;"		:'Î',
                "&Iuml;"		:'Ï',
                "&ETH;"			:'Ð',
                "&Ntilde;"		:'Ñ',
                "&Ograve;"		:'Ò',
                "&Oacute;"		:'Ó',
                "&Ocirc;"		:'Ô',
                "&Otilde;"		:'Õ',
                "&Ouml;"		:'Ö',
                "&times;"		:'×',
                "&Oslash;"		:'Ø',
                "&Ugrave;"		:'Ù',
                "&Uacute;"		:'Ú',
                "&Ucirc;"		:'Û',
                "&Uuml;"		:'Ü',
                "&Yacute;"		:'Ý',
                "&THORN;"		:'Þ',
                "&szlig;"		:'ß',
                "&agr;"			:'à',
                "&agrave;"		:'à',
                "&aacute;"		:'á',
                "&acirc;"		:'â',
                "&atilde;"		:'ã',
                "&auml;"		:'ä',
                "&aring;"		:'å',
                "&aelig;"		:'æ',
                "&ccedil;"		:'ç',
                "&egrave;"		:'è',
                "&eacute;"		:'é',
                "&ecirc;"		:'ê',
                "&euml;"		:'ë',
                "&igrave;"		:'ì',
                "&iacute;"		:'í',
                "&icirc;"		:'î',
                "&iuml;"		:'ï',
                "&eth;"			:'ð',
                "&ntilde;"		:'ñ',
                "&ograve;"		:'ò',
                "&oacute;"		:'ó',
                "&ocirc;"		:'ô',
                "&otilde;"		:'õ',
                "&ouml;"		:'ö',
                "&divide;"		:'÷',
                "&oslash;"		:'ø',
                "&ugrave;"		:'ù',
                "&uacute;"		:'ú',
                "&ucirc;"		:'û',
                "&uuml;"		:'ü',
                "&yacute;"		:'ý',
                "&thorn;"		:'þ',
                "&yuml;"		:'ÿ',

                # HTML Entities Table - Symbols and Greek characters
                "&fnof;"		:"ƒ",
                "&Alpha;"		:"Α",
                "&Beta;"		:"Β",
                "&Gamma;"		:"Γ",
                "&Delta;"		:"Δ",
                "&Epsilon;"		:"Ε",
                "&Zeta;"		:"Ζ",
                "&Eta;"			:"Η",
                "&Theta;"		:"Θ",
                "&Iota;"		:"Ι",
                "&Kappa;"		:"Κ",
                "&Lambda;"		:"Λ",
                "&Mu;"			:"Μ",
                "&Nu;"			:"Ν",
                "&Xi;"			:"Ξ",
                "&Omicron;"		:"Ο",
                "&Pi;"			:"Π",
                "&Rho;"			:"Ρ",
                "&Sigma;"		:"Σ",
                "&Tau;"			:"Τ",
                "&Upsilon;"		:"Υ",
                "&Phi;"			:"Φ",
                "&Chi;"			:"Χ",
                "&Psi;"			:"Ψ",
                "&Omega;"		:"Ω",
                "&alpha;"		:"α",
                "&beta;"		:"β",
                "&gamma;"		:"γ",
                "&delta;"		:"δ",
                "&epsilon;"		:"ε",
                "&zeta;"		:"ζ",
                "&eta;"			:"η",
                "&theta;"		:"θ",
                "&iota;"		:"ι",
                "&kappa;"		:"κ",
                "&lambda;"		:"λ",
                "&mu;"			:"μ",
                "&nu;"			:"ν",
                "&xi;"			:"ξ",
                "&omicron;"		:"ο",
                "&pi;"			:"π",
                "&rho;"			:"ρ",
                "&sigmaf;"		:"ς",
                "&sigma;"		:"σ",
                "&tau;"			:"τ",
                "&upsilon;"		:"υ",
                "&phi;"			:"φ",
                "&chi;"			:"χ",
                "&psi;"			:"ψ",
                "&omega;"		:"ω",
                "&thetasym;"	:"ϑ",
                "&upsih;"		:"ϒ",
                "&piv;"			:"ϖ",
                "&bull;"		:"•",
                "&hellip;"		:"…",
                "&prime;"		:"′",
                "&Prime;"		:"″",
                "&oline;"		:"‾",
                "&frasl;"		:"⁄",
                "&weierp;"		:"℘",
                "&image;"		:"ℑ",
                "&real;"		:"ℜ",
                "&trade;"		:"™",
                "&alefsym;"		:"ℵ",
                "&larr;"		:"←",
                "&uarr;"		:"↑",
                "&rarr;"		:"→",
                "&darr;"		:"↓",
                "&harr;"		:"↔",
                "&crarr;"		:"↵",
                "&lArr;"		:"⇐",
                "&uArr;"		:"⇑",
                "&rArr;"		:"⇒",
                "&dArr;"		:"⇓",
                "&hArr;"		:"⇔",
                "&forall;"		:"∀",
                "&part;"		:"∂",
                "&exist;"		:"∃",
                "&empty;"		:"∅",
                "&nabla;"		:"∇",
                "&isin;"		:"∈",
                "&notin;"		:"∉",
                "&ni;"			:"∋",
                "&prod;"		:"∏",
                "&sum;"			:"∑",
                "&minus;"		:"−",
                "&lowast;"		:"∗",
                "&radic;"		:"√",
                "&prop;"		:"∝",
                "&infin;"		:"∞",
                "&ang;"			:"∠",
                "&and;"			:"∧",
                "&or;"			:"∨",
                "&cap;"			:"∩",
                "&cup;"			:"∪",
                "&int;"			:"∫",
                "&there4;"		:"∴",
                "&sim;"			:"∼",
                "&cong;"		:"≅",
                "&asymp;"		:"≈",
                "&ne;"			:"≠",
                "&equiv;"		:"≡",
                "&le;"			:"≤",
                "&ge;"			:"≥",
                "&sub;"			:"⊂",
                "&sup;"			:"⊃",
                "&nsub;"		:"⊄",
                "&sube;"		:"⊆",
                "&supe;"		:"⊇",
                "&oplus;"		:"⊕",
                "&otimes;"		:"⊗",
                "&perp;"		:"⊥",
                "&sdot;"		:"⋅",
                "&lceil;"		:"⌈",
                "&rceil;"		:"⌉",
                "&lfloor;"		:"⌊",
                "&rfloor;"		:"⌋",
                "&lang;"		:"⟨",
                "&rang;"		:"⟩",
                "&loz;"			:"◊",
                "&spades;"		:"♠",
                "&clubs;"		:"♣",
                "&hearts;"		:"♥",
                "&diams;"		:"♦",

                # HTML Entities Table - Special characters
                "&quot;"		:"\"",
                "&amp;"			:"&",
                "&lt;"			:"<",
                "&gt;"			:">",
                "&OElig;"		:"Œ",
                "&oelig;"		:"œ",
                "&Scaron;"		:"Š",
                "&scaron;"		:"š",
                "&Yuml;"		:"Ÿ",
                "&circ;"		:"ˆ",
                "&tilde;"		:"˜",
                "&ensp;"		:" ",
                "&emsp;"		:" ",
                "&thinsp;"		:" ",
                "&zwnj;"		:"‌",
                "&zwj;"			:"‍",
                "&lrm;"			:"‎",
                "&rlm;"			:"‏",
                "&ndash;"		:"–",
                "&mdash;"		:"—",
                "&lsquo;"		:"‘",
                "&rsquo;"		:"’",
                "&sbquo;"		:"‚",
                "&ldquo;"		:"“",
                "&rdquo;"		:"”",
                "&bdquo;"		:"„",
                "&dagger;"		:"†",
                "&Dagger;"		:"‡",
                "&permil;"		:"‰",
                "&lsaquo;"		:"‹",
                "&rsaquo;"		:"›",
                "&euro;"		:"€",
                
            # https://symbl.cc/en/html-entities/
            # todo

            # https://www.ams.org/STIX/
            # https://www.ams.org/STIX/bnb/stix-tbl.ascii-2006-10-20
                "&Agr;"			:"\u0391",
                "&Bgr;"			:"\u0392",
                "&Ggr;"			:"\u0393",
                "&Dgr;"			:"\u0394",
                "&Egr;"			:"\u0395",
                "&Zgr;"			:"\u0396",
                "&EEgr;"		:"\u0397",
                "&THgr;"		:"\u0398",
                "&Igr;"			:"\u0399",
                "&Kgr;"			:"\u039A",
                "&Lgr;"			:"\u039B",
                "&Mgr;"			:"\u039C",
                "&Ngr;"			:"\u039D",
                "&Ygr;"			:"\u039E",
                "&Ogr;"			:"\u039F",
                "&Pgr;"			:"\u03A0",
                "&Rgr;"			:"\u03A1",
                "&Sgr;"			:"\u03A3",
                "&Tgr;"			:"\u03A4",
                "&Fgr;"			:"\u03A6",
                "&KHgr;"		:"\u03A7",
                "&PSgr;"		:"\u03A8",
                "&OHgr;"		:"\u03A9",
                "&agr;"			:"\u03B1",
                "&bgr;"			:"\u03B2",
                "&ggr;"			:"\u03B3",
                "&dgr;"			:"\u03B4",
                "&Vegr;"		:"\u03B5",
                "&zgr;"			:"\u03B6",
                "&eegr;"		:"\u03B7",
                "&thgr;"		:"\u03B8",
                "&ptheta;"		:"\u03B8",
                "&igr;"			:"\u03B9",
                "&kgr;"			:"\u03BA",
                "&lgr;"			:"\u03BB",
                "&mgr;"			:"\u03BC",
                "&ngr;"			:"\u03BD",
                "&xgr;"			:"\u03BE",
                "&ogr;"			:"\u03BF",
                "&pgr;"			:"\u03C0",
                "&rgr;"			:"\u03C1",
                "&sfgr;"		:"\u03C2",
                "&sgr;"			:"\u03C3",
                "&tgr;"			:"\u03C4",
                "&ugr;"			:"\u03C5",
                "&Jgr;"			:"\u03C6",
                "&khgr;"		:"\u03C7",
                "&psgr;"		:"\u03C8",
                "&ohgr;"		:"\u03C9",
                "&Vbgr;"		:"\u03D0",
                "&Vthgr;"		:"\u03D1",
                "&Ugr;"			:"\u03D2",
                "&fgr;"			:"\u03D5",
                "&Vpgr;"		:"\u03D6",
                "&diag;"		:"\u03DD",
                "&Vkgr;"		:"\u03F0",
                "&Vrgr;"		:"\u03F1",
                "&Thgr;"		:"\u03F4",
                "&egr;"			:"\u03F5",
                "&egrb;"		:"\u03F6",
}

matsciBERT_mapping = {
                "ɛ":"∈",
                "∈":"∈",
                "Ɛ":"∈",
                "ⅇ":"e",
                "ƒ":"f",
                "⇌":"⇋",
                "ⅰ":"i",
                "。":".",
                "k":"k",
                "ξ":"ξ",
                "Ξ":"ξ",
                "−":"-",
                "p":"p",
                "‒":"-",
                ">":">",
                "～":"~",
                "?":"?",
                "в":"в",
                "⊃":"⊃",
                "[":"[",
                "6":"6",
                "ℋ":"H",
                "‐":"-",
                "и":"и",
                "ℒ":"L",
                "đ":"d",
                "⟨":"<",
                "o":"o",
                "ⅆ":"d",
                "○":"*",
                "ᴼ":"°",
                "∘":"°",
                "4":"4",
                "‧":".",
                "ǀ":"|",
                "˄":"^",
                "ɸ":"φ",
                "˃":">",
                "⁰":"°",
                "⇄":"⇋",
                "•":"•",
                "\\":"\\",
                "ℳ":"M",
                "р":"p",
                "⑥":"6",
                "˚":"°",
                "⁓":"~",
                "d":"d",
                "②":"2",
                "ʈ":"t",
                "v":"v",
                "↦":"→",
                "s":"s",
                "¨":"\"",
                "ˆ":"^",
                "⇔":"⇔",
                "︸":"︸",
                "¯":"-",
                "ο":"o",
                "”":"\"",
                "x":"x",
                "³":"3",
                "n":"n",
                "˜":"~",
                "µ":"μ",
                "│":"|",
                "ƭ":"t",
                "g":"g",
                "t":"t",
                "^":"^",
                "①":"1",
                "⑤":"5",
                "⇐":"⇐",
                "r":"r",
                "˛":",",
                "¸":",",
                "‰":"%",
                "0":"0",
                "↑":"↑",
                "▽":"▽",
                "ι":"ι",
                "#":"#",
                "＃":"#",
                "ѵ":"v",
                "‑":"-",
                "‹":"<",
                "’":"'",
                "％":"%",
                "о":"o",
                "ℙ":"P",
                "\"":"\"",
                "у":"y",
                "1":"1",
                "˗":"-",
                "∕":"/",
                "ø":"∅",
                "Ø":"∅",
                "◯":"O",
                "j":"j",
                "۰":".",
                "⁺":"+",
                "]":"]",
                "u":"u",
                "ф":"φ",
                "Ф":"φ",
                "τ":"τ",
                "₄":"4",
                "⑧":"8",
                "7":"7",
                "˂":"<",
                "%":"%",
                "₈":"8",
                "∖":"\\",
                "ı":"ı",
                "›":">",
                "$":"$",
                "→":"→",
                "{":"{",
                "⁄":"/",
                "ρ":"ρ",
                "ϱ":"ρ",
                "υ":"υ",
                "γ":"γ",
                "ɣ":"γ",
                "Ɣ":"γ",
                "ϒ":"γ",
                "²":"2",
                "÷":"÷",
                "ℝ":"R",
                "·":".",
                "9":"9",
                "д":"d",
                "↣":"→",
                "ᵒ":"°",
                "ℛ":"R",
                "*":"*",
                "@":"@",
                "w":"w",
                "с":"c",
                "ε":"ε",
                ".":".",
                "↓":"↓",
                "i":"i",
                "z":"z",
                "/":"/",
                "y":"y",
                "×":"×",
                "l":"l",
                "ℬ":"B",
                "ƞ":"η",
                "Ƞ":"η",
                "“":"\"",
                "-":"-",
                "←":"←",
                "）":")",
                "ג":"λ",
                "④":"4",
                ")":")",
                "↔":"⇔",
                "∼":"~",
                "}":"}",
                "і":"i",
                "＞":">",
                "ℂ":"C",
                "¦":"|",
                "＜":"<",
                "ϰ":"κ",
                "，":",",
                "ƛ":"λ",
                "⟦":"[",
                "–":"-",
                "₇":"7",
                "¬":"¬",
                "q":"q",
                "ⓒ":"©",
                "ζ":"ζ",
                "－":"-",
                "﹥":">",
                "˭":"=",
                "e":"e",
                "&":"&",
                "¡":"!",
                "﹤":"<",
                "◦":"°",
                "║":"|",
                "〉":">",
                "′":"'",
                "₆":"6",
                "h":"h",
                "׀":"|",
                "ˉ":"-",
                "⟩":">",
                "к":"κ",
                "|":"|",
                "!":"!",
                "f":"f",
                "＆":"&",
                "ʌ":"^",
                "﹒":".",
                "_":"_",
                "〈":"<",
                "¶":"¶",
                "ℰ":"E",
                "˝":"\"",
                ";":";",
                "⁎":"*",
                "ĸ":"κ",
                "ϲ":"c",
                "ɳ":"η",
                "ω":"ω",
                "ₒ":"0",
                "е":"e",
                "⇆":"⇋",
                "⇋":"⇋",
                "ϐ":"β",
                "₃":"3",
                ":":":",
                "⑦":"7",
                "♯":"#",
                "‚":",",
                "2":"2",
                "：":":",
                "ℜ":"R",
                "ħ":"h",
                "3":"3",
                "м":"м",
                "ʼ":"'",
                "ᴓ":"∅",
                "c":"c",
                "†":"+",
                "⸳":".",
                "；":";",
                "μ":"μ",
                "ʺ":"\"",
                "(":"(",
                "∗":"∗",
                "‘":"‘",
                "（":"(",
                "ð":"δ",
                "~":"~",
                "˙":"°",
                "º":"°",
                "a":"a",
                "m":"m",
                "б":"6",
                "+":"+",
                "‾":"-",
                "b":"b",
                "ʻ":"`",
                "ⓡ":"r",
                "Ⓡ":"r",
                "⇒":"→",
                "∣":"|",
                "5":"5",
                ",":",",
                "κ":"κ",
                "═":"=",
                "ʗ":"C",
                "8":"8",
                "∽":"~",
                "π":"π",
                "ϖ":"π",
                "Π":"π",
                "η":"η",
                "т":"τ",
                "δ":"δ",
                "<":"<",
                "⨍":"f",
                "'":"'",
                "Þ":"p",
                "＋":"+",
                "п":"п",
                "λ":"λ",
                "Λ":"λ",
                "③":"3",
                "β":"β",
                "α":"α",
                "ν":"ν",
                "`":"`",
                "●":"●",
                "✓":"●",
                "▪":"●",
                "➢":"●",
                "➣":"●",
                "■":"●",
                "❖":"●",
                "♦":"●",
                "а":"a",
                "ł":"l",
                "⧸":"/",
                "─":"-",
                "ʹ":"'",
                "σ":"σ",
                "ς":"σ",
                "´":"'",
                "―":"―",
                "χ":"χ",
                "∧":"∧",
                "—":"—",
                "⊂":"⊂",
                "«":"«",
                "»":"»",
                "≥":"≥",
                "‖":"‖",
                "ψ":"ψ",
                "≡":"≡",
                "⋃":"⋃",
                "¢":"¢",
                "∝":"∝",
                "ß":"β",
                "⊕":"⊕",
                "∨":"∨",
                "∞":"∞",
                "∆":"∆",
                "▵":"∆",
                "⊿":"∆",
                "∪":"∪",
                "⊆":"⊆",
                "⊥":"⊥",
                "∙":"∙",
                "∑":"∑",
                "∀":"∀",
                "∇":"∇",
                "„":"„",
                "±":"±",
                "≃":"≃",
                "æ":"æ",
                "√":"√",
                "®":"®",
                "∂":"∂",
                "£":"£",
                "≈":"≈",
                "∏":"∏",
                "‡":"‡",
                "∅":"∅",
                "≪":"≪",
                "⋆":"⋆",
                "∫":"∫",
                "ʃ":"∫",
                "∮":"∮",
                "⊗":"⊗",
                "€":"€",
                "∈":"∈",
                "∥":"∥",
                "θ":"θ",
                "Ө":"θ",
                "ɵ":"θ",
                "Ɵ":"θ",
                "ϴ":"θ",
                "Θ":"θ",
                "φ":"φ",
                "ʋ":"ʋ",
                "≺":"≺",
                "°":"°",
                "≤":"≤",
                "⪡":"≤",
                "∩":"∩",
                "©":"©",
                "þ":"þ",
                "ﬂ":"fl",
                "§":"§",
                "¥":"¥",
                "∃":"∃",
                "⋅":"⋅",
                "⌀":"φ",
                "≔":":=",
                "⊤":"T",
                "⪆":"≥",
                "⨯":"×",
                "≊":"≃",
                "≽":"≥",
                "΄":"'",
                "✕":"×",
                "⪯":"≤",
                "ӏ":"I",
                "≲":"≤",
                "≫":">>",
                "≅":"≃",
                "ԑ":"∈",
                "➔":"→",
                "∭":"∫∫∫",
                "☆":"⋆",
                "≻":">",
                "⟧":"]]",
                "⋘":"<<<",
                "⩾":"≥",
                "⨂":"⊗",
                "⬄":"⇔",
                "⟷":"⇔",
                "⟺":"⇔",
                "⍵":"ω",
                "Ω":"Ω",
                "℃":"°C",
                "≳":"≥",
                "⋙":">>>",
                "⩽":"≤",
                "⊖":"θ",
                "ϑ":"θ",
                "⪰":"≥",
                "⟶":"→",
                "≿":"≥",
                "⨉":"×",
                "⟵":"←",
                "ϵ":"∈",
                "≌":"≃",
                "≾":"≤",
                "∬":"∫∫",
                "☓":"×",
                "⊝":"θ",
                "∓":"±",
                "★":"⋆",
                "ẟ":"δ",
                "△":"Δ",
                "Δ":"Δ",
                "∊":"ε",
                "ℏ":"ћ",
                "ћ":"ћ",
                "⪕":"≤",
                "ℓ":"l",
                "⪢":">",
                "∷":"::",
                "⪅":"≤",
                "₂":"2",
                "⃞":"]",
                "׳":"'",
                "⍺":"α",
                "ɑ":"α",
                "⋍":"≃",
                "⪝":"≤",
                "〞":"\'\'",
                "″":"\'\'",
                "ϭ":"6",
                "￥":"¥",
                "½":"1/2",
                "⁃":"--",
                "》":">>",
                "⅚":"5/6",
                "ⅳ":"iv",
                "х":"x",
                "ⅲ":"iii",
                "ﬁ":"fi",
                "ⅱ":"ii",
                "…":"...",
                "ŋ":"η",
                "¾":"3/4",
                "ѕ":"s",
                "ϕ":"∅",
                "Φ":"φ",
                "㎛":"μm",
                "〚":"[[",
                "ⅵ":"vi",
                "™":"(tm)",
                "ǁ":"||",
                "‴":"\'\'\'",
                "ѱ":"ψ",
                "Ψ":"ψ",
                "〛":"]]",
                "⁗":"\'\'\'\'",
                "㎜":"mm",
                "㎚":"nm",
                "⅔":"2/3",
                "≧":"≥",
                "ﬀ":"ff",
                "є":"∈",
                "Є":"∈",
                "¼":"1/4",
                "⅓":"1/3",
                "⅜":" 3/8", 
                "≦":"≤",
                "⪖":"≥",
                "Г":"Г",
                "Γ":"Г",
                "=":"=",
                "A":"A",
                "B":"B",
                "C":"C",
                "D":"D",
                "E":"E",
                "F":"F",
                "G":"G",
                "H":"H",
                "н":"H",
                "I":"I",
                "J":"J",
                "K":"K",
                "L":"L",
                "M":"M",
                "N":"N",
                "O":"O",
                "О":"O",
                "P":"P",
                "Q":"Q",
                "R":"R",
                "S":"S",
                "T":"T",
                "U":"U",
                "V":"V",
                "Ѵ":"V",
                "W":"W",
                "X":"X",
                "Y":"Y",
                "Z":"Z",
                "Σ":"Σ",
                "Ʃ":"Σ",
                "Т":"T",
                "Τ":"T",
                "Α":"A",
                "М":"M",
                "Ο":"O",
                "А":"A",
                "В":"B",
                "Ⅰ":"I",
                "Е":"E",
                "Н":"H",
                "Ν":"N",
                "Ϲ":"C",
                "Ⅱ":"II",
                "Ⅲ":"III",
                "Ⅳ":"IV",
                "Ⅵ":"VI",
                "Ⅶ":"VII",
                "Ⅷ":"VIII",
                "Η":"H",
                "ﬃ":"ffi",
                "Υ":"Y",
                "Μ":"M",
                "Β":"B",
                "Ү":"Y",
                "Χ":"X",
                "Ε":"E",
                "Ł":"L",
                "Ρ":"P",
                "Κ":"K",
                "К":"K",
                "Ӏ":"I",
                "С":"C",
                "Ζ":"Z",
                "Р":"Р",
                "Х":"X",
                "Ι":"I",
                "І":"I",
                "Ѕ":"S",
                "Б":"6",
                "⿿":"-",
                "⬜":"-",
                "⋯":"--",
                "∠":"∠",
                "□":"□",
                "㏑":"ln",
                "⇀":"⇀",
                "⋮":" ",
                "␣":" ",
                "⌢":" ",
                "ℵ":"N",
                "、":",",
                "ˇ":"ˇ",
                "л":"π",
                "⌊":"[",
                "⌋":"]",
                "┴":"⊥",
                "⊙":"⊙",
                "П":"II",
                "≒":"~",
                "⧹":"/",
                "№":"N",
                "⋱":" ",
                "ˊ":"'",
                "⟹":"→",
                "⥄":" ",
                "Ʌ":"λ",
                "↕":" ",
                "Ш":"III",
                "▿":"∇",
                "℘":"P",
                "я":" ",
                "▒":" ",
                "▴":"Δ",
                "ә":"ə",
                "ə":"ə",
                "Ѱ":"ψ",
                "⏟":"︸",
                "г":" ",
                "œ":" ",
                "▹":" ",
                "◄":" ",
                "⁻":"-",
                "∴":"∴",
                "⊺":" ",
                "︷":"︷",
                "˘":" ",
                "И":" ",
                "↙":"-",
                "≜":"≜",
                "⌣":" ",
                "ы":" ",
                "Ͽ":" ",
                "⌈":"[",
                "⌉":"]",
                "◊":"●",
                "▾":"●",
                "◆":"●",
                "∎":"●",
                "∶":":",
                "ь":" ",
                "ˑ":".",
                "・":".",
                "ɷ":"ω",
                "ⅹ":"*",
                "Ⅴ":"V",
                "Ʈ":"τ",
                "╱":" ",
                "╲":" ",
                "＝":"=",
}


#: Control characters.
CONTROLS = {
    '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u000e', '\u000f', '\u0011',
    '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018', '\u0019', '\u001a', '\u001b',
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    '-',  # \u002d Hyphen-minus
    '‐',  # \u2010 Hyphen
    '‑',  # \u2011 Non-breaking hyphen
    '⁃',  # \u2043 Hyphen bullet
    '‒',  # \u2012 figure dash
    '–',  # \u2013 en dash
    '—',  # \u2014 em dash
    '―',  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    '-',  # \u002d Hyphen-minus
    '−',  # \u2212 Minus
    '－',  # \uff0d Full-width Hyphen-minus
    '⁻',  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    '+',  # \u002b Plus
    '＋',  # \uff0b Full-width Plus
    '⁺',  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    '/',  # \u002f Solidus
    '⁄',  # \u2044 Fraction slash
    '∕',  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    '~',  # \u007e Tilde
    '˜',  # \u02dc Small tilde
    '⁓',  # \u2053 Swung dash
    '∼',  # \u223c Tilde operator
    '∽',  # \u223d Reversed tilde
    '∿',  # \u223f Sine wave
    '〜',  # \u301c Wave dash
    '～',  # \uff5e Full-width tilde
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    '’',  # \u2019
    '՚',  # \u055a
    'Ꞌ',  # \ua78b
    'ꞌ',  # \ua78c
    '＇',  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    '‘',  # \u2018
    '’',  # \u2019
    '‚',  # \u201a
    '‛',  # \u201b

}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    '“',  # \u201c
    '”',  # \u201d
    '„',  # \u201e
    '‟',  # \u201f
}

#: Accent characters.
ACCENTS = {
    '`',  # \u0060
    '´',  # \u00b4
}

#: Prime characters.
PRIMES = {
    '′',  # \u2032
    '″',  # \u2033
    '‴',  # \u2034
    '‵',  # \u2035
    '‶',  # \u2036
    '‷',  # \u2037
    '⁗',  # \u2057
}

CHAR_REPLACEMENTS = [
    ('\[?\[1 with combining macron\]\]?', '1\u0304'),
    ('\[?\[2 with combining macron\]\]?', '2\u0304'),
    ('\[?\[3 with combining macron\]\]?', '3\u0304'),
    ('\[?\[4 with combining macron\]\]?', '4\u0304'),
    ('\[?\[approximate\]\]?', '\u2248'),
    ('\[?\[bottom\]\]?', '\u22a5'),
    ('\[?\[c with combining tilde\]\]?', 'C\u0303'),
    ('\[?\[capital delta\]\]?', '\u0394'),
    ('\[?\[capital lambda\]\]?', '\u039b'),
    ('\[?\[capital omega\]\]?', '\u03a9'),
    ('\[?\[capital phi\]\]?', '\u03a6'),
    ('\[?\[capital pi\]\]?', '\u03a0'),
    ('\[?\[capital psi\]\]?', '\u03a8'),
    ('\[?\[capital sigma\]\]?', '\u03a3'),
    ('\[?\[caret\]\]?', '^'),
    ('\[?\[congruent with\]\]?', '\u2245'),
    ('\[?\[curly or open phi\]\]?', '\u03d5'),
    ('\[?\[dagger\]\]?', '\u2020'),
    ('\[?\[dbl greater-than\]\]?', '\u226b'),
    ('\[?\[dbl vertical bar\]\]?', '\u2016'),
    ('\[?\[degree\]\]?', '\xb0'),
    ('\[?\[double bond, length as m-dash\]\]?', '='),
    ('\[?\[double bond, length half m-dash\]\]?', '='),
    ('\[?\[double dagger\]\]?', '\u2021'),
    ('\[?\[double equals\]\]?', '\u2267'),
    ('\[?\[double less-than\]\]?', '\u226a'),
    ('\[?\[double prime\]\]?', '\u2033'),
    ('\[?\[downward arrow\]\]?', '\u2193'),
    ('\[?\[fraction five-over-two\]\]?', '5/2'),
    ('\[?\[fraction three-over-two\]\]?', '3/2'),
    ('\[?\[gamma\]\]?', '\u03b3'),
    ('\[?\[greater-than-or-equal\]\]?', '\u2265'),
    ('\[?\[greater, similar\]\]?', '\u2273'),
    ('\[?\[gt-or-equal\]\]?', '\u2265'),
    ('\[?\[i without dot\]\]?', '\u0131'),
    ('\[?\[identical with\]\]?', '\u2261'),
    ('\[?\[infinity\]\]?', '\u221e'),
    ('\[?\[intersection\]\]?', '\u2229'),
    ('\[?\[iota\]\]?', '\u03b9'),
    ('\[?\[is proportional to\]\]?', '\u221d'),
    ('\[?\[leftrightarrow\]\]?', '\u2194'),
    ('\[?\[leftrightarrows\]\]?', '\u21c4'),
    ('\[?\[less-than-or-equal\]\]?', '\u2264'),
    ('\[?\[less, similar\]\]?', '\u2272'),
    ('\[?\[logical and\]\]?', '\u2227'),
    ('\[?\[middle dot\]\]?', '\xb7'),
    ('\[?\[not equal\]\]?', '\u2260'),
    ('\[?\[parallel\]\]?', '\u2225'),
    ('\[?\[per thousand\]\]?', '\u2030'),
    ('\[?\[prime or minute\]\]?', '\u2032'),
    ('\[?\[quadruple bond, length as m-dash\]\]?', '\u2263'),
    ('\[?\[radical dot\]\]?', ' \u0307'),
    ('\[?\[ratio\]\]?', '\u2236'),
    ('\[?\[registered sign\]\]?', '\xae'),
    ('\[?\[reverse similar\]\]?', '\u223d'),
    ('\[?\[right left arrows\]\]?', '\u21C4'),
    ('\[?\[right left harpoons\]\]?', '\u21cc'),
    ('\[?\[rightward arrow\]\]?', '\u2192'),
    ('\[?\[round bullet, filled\]\]?', '\u2022'),
    ('\[?\[sigma\]\]?', '\u03c3'),
    ('\[?\[similar\]\]?', '\u223c'),
    ('\[?\[small alpha\]\]?', '\u03b1'),
    ('\[?\[small beta\]\]?', '\u03b2'),
    ('\[?\[small chi\]\]?', '\u03c7'),
    ('\[?\[small delta\]\]?', '\u03b4'),
    ('\[?\[small eta\]\]?', '\u03b7'),
    ('\[?\[small gamma, Greek, dot above\]\]?', '\u03b3\u0307'),
    ('\[?\[small kappa\]\]?', '\u03ba'),
    ('\[?\[small lambda\]\]?', '\u03bb'),
    ('\[?\[small micro\]\]?', '\xb5'),
    ('\[?\[small mu \]\]?', '\u03bc'),
    ('\[?\[small nu\]\]?', '\u03bd'),
    ('\[?\[small omega\]\]?', '\u03c9'),
    ('\[?\[small phi\]\]?', '\u03c6'),
    ('\[?\[small pi\]\]?', '\u03c0'),
    ('\[?\[small psi\]\]?', '\u03c8'),
    ('\[?\[small tau\]\]?', '\u03c4'),
    ('\[?\[small theta\]\]?', '\u03b8'),
    ('\[?\[small upsilon\]\]?', '\u03c5'),
    ('\[?\[small xi\]\]?', '\u03be'),
    ('\[?\[small zeta\]\]?', '\u03b6'),
    ('\[?\[space\]\]?', ' '),
    ('\[?\[square\]\]?', '\u25a1'),
    ('\[?\[subset or is implied by\]\]?', '\u2282'),
    ('\[?\[summation operator\]\]?', '\u2211'),
    ('\[?\[times\]\]?', '\xd7'),
    ('\[?\[trade mark sign\]\]?', '\u2122'),
    ('\[?\[triple bond, length as m-dash\]\]?', '\u2261'),
    ('\[?\[triple bond, length half m-dash\]\]?', '\u2261'),
    ('\[?\[triple prime\]\]?', '\u2034'),
    ('\[?\[upper bond 1 end\]\]?', ''),
    ('\[?\[upper bond 1 start\]\]?', ''),
    ('\[?\[upward arrow\]\]?', '\u2191'),
    ('\[?\[varepsilon\]\]?', '\u03b5'),
    ('\[?\[x with combining tilde\]\]?', 'X\u0303'),
]