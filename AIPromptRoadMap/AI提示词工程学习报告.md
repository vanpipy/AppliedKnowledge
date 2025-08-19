

# **AI提示词工程系统性掌握指南**

**引言**

提示词工程，作为引导生成式人工智能（Generative AI）解决方案生成所需输出的核心实践，正日益成为人工智能领域不可或缺的技能。它不仅仅是简单地输入文字，更涉及精心选择最合适的格式、短语、词汇和符号，以指导AI模型更有效地与用户交互 1。从根本上讲，提示词工程是连接人类意图与机器理解的桥梁，通过精确的输入来塑造AI的行为和输出 3。

这项技能的重要性体现在多个层面。首先，它能够显著增强开发者对AI交互的控制。有效的提示词为大型语言模型（LLM）提供了明确的意图和丰富的上下文，这使得AI模型能够更精准地提炼信息，并以所需的格式呈现输出。这种精确性还有助于防止用户滥用AI或请求AI无法准确处理的内容，例如在商业应用中限制生成不当内容 1。其次，提示词工程极大地改善了用户体验。通过精心设计的提示词，用户可以避免耗时的反复试错，从而在首次交互中就能从AI工具获得连贯、准确且高度相关的响应。此外，它还能在一定程度上减轻LLM训练数据中可能存在的偏差，并增强AI对用户意图的理解，即使初始输入信息有限 1。

更进一步，提示词工程提高了AI应用的灵活性与可扩展性。提示工程师能够创建包含领域中立指令的提示词和模板，这些资源可以在企业内部快速复用，从而以规模化方式扩展AI投资。例如，一个用于流程优化的通用提示词可以应用于不同的业务流程和部门，极大地提高了效率 1。同时，精细调整提示词可以显著提高模型输出的质量和相关性，从而减少对人工审查和后生成编辑的需求，节省时间和精力，并降低模型运行成本 2。

提示词工程的演变反映了AI模型能力的指数级增长，从简单的指令遵循到复杂的推理和多模态交互。早期AI模型的能力相对有限，因此提示词设计也较为直接，主要用于引导模型执行简单的命令 3。然而，随着大型语言模型（LLM）的规模不断扩大，它们开始展现出“涌现能力”，例如通过思维链（Chain-of-Thought）进行复杂的多步推理 4。这种模型能力的飞跃直接推动了提示词工程从最初的“简单命令”层面，迅速发展到需要设计复杂策略来激发模型深层推理能力的阶段 3。当模型能够处理并理解图像、音频等多种模态的输入时，提示词工程的边界也随之拓宽，需要整合多模态信息，从而将提示词工程从纯粹的文本领域推向更广阔的“体验设计”和“工作流编排”领域 5。这种演变路径清晰地表明，提示词工程的发展与AI模型自身能力的进步是紧密耦合、相互促进的。

提示词工程的有效性直接影响AI应用的商业价值和用户满意度，是AI产品成功的关键因素。AI模型的输出质量是其应用价值的直接体现。如果提示词设计不当，导致AI模型生成模糊、不准确或不相关的输出 1，这将直接损害用户体验，降低用户对AI工具的信任和满意度 1。用户会因为无法获得预期结果而放弃使用，从而使得AI解决方案无法发挥其应有的商业价值。反之，精心设计、高度优化的提示词能够确保AI模型提供精确、相关且符合预期的响应，显著提升用户满意度，进而推动AI产品被广泛采用，并最终实现其商业目标 1。因此，提示词工程不仅仅是技术层面的优化，更是产品设计和市场成功的核心驱动力。

展望未来，自动化和代理化是提示词工程的必然发展方向，以应对大规模部署和复杂场景下的挑战。在AI应用规模化部署到企业级系统时，手动创建和优化提示词变得极其低效且难以维护。模型、数据和业务需求的快速变化使得静态提示词无法适应。为了解决这种可扩展性和动态适应性的瓶颈，研究和实践自然而然地转向了自动化方法，例如自动化提示词优化（APO）和ReAct prompting 5。此外，将提示词视为AI基础设施中能够实时演进的“活的组件”，并将其整合到由多个AI代理组成的复杂工作流中，这预示着提示词工程将从战术性实践转变为产品开发和企业自动化的核心支柱 5。

**第一部分 \- 系统性学习路线图：AI提示词工程从基础到高级**

### **1\. 基础入门**

掌握提示词工程，首先需要对大型语言模型（LLM）的基本工作原理有一个清晰的认识，并理解基础的提示词设计原则。

#### **1.1. 理解大型语言模型 (LLM) 工作原理**

LLM是提示词工程的基础，了解其内部机制有助于更有效地设计提示词。

* Tokens (令牌)  
  令牌是LLM处理文本的基本单位，可以是单个单词、字符集，甚至是词语和标点符号的组合 10。在模型训练过程中，输入的文本会被分解成这些离散的令牌。LLM通过分析这些令牌之间的语义关系，例如它们共同出现的频率或在相似上下文中的使用方式，来学习并最终生成输出序列 11。理解令牌的粒度对于控制模型的输入和输出长度至关重要，因为不同的分词方法（如词分词、字符分词或子词分词）会影响给定文本被分解成的令牌数量，进而影响模型的计算资源消耗和最大处理能力 11。  
* Context Window (上下文窗口)  
  上下文窗口，又称“上下文长度”，是指LLM在任何给定时间可以考虑或“记住”的文本量，其单位也是令牌 12。可以将其视为LLM的“工作记忆”。一个更大的上下文窗口意味着模型能够处理更长的输入，并在每次输出中整合更多的信息。这直接决定了模型在长对话中保持连贯性以及一次性处理大型文档或代码样本的能力 13。研究表明，LLM在处理长上下文时，对信息的位置敏感，通常在输入上下文的开头或结尾处表现最佳，而中间部分的信息处理性能可能会下降 13。因此，有效利用上下文窗口，例如将最相关的信息置于提示词的关键位置，是优化模型性能的重要考量。  
* 模型参数 (Temperature, Top-P, Max Length, Stop Sequences, Penalties)  
  除了文本输入，LLM还提供了一系列可配置的参数，这些参数能够微调模型的行为，从而影响其生成输出的特性。  
  * **Temperature (温度)**：这个参数调节模型输出的随机性。较高的温度设置会使模型在选择下一个令牌时更倾向于选择概率较低的选项，从而产生更具创造性和不可预测性的输出。相反，较低的温度会使模型更保守，倾向于选择概率最高的令牌，导致结果更可预测和确定性 14。在需要发散性思维（如创意写作）时，可以提高温度；在需要精确事实（如问答）时，则应降低温度。  
  * **Top-P (核采样)**：Top-P是另一种控制模型随机性的参数，它通过设定一个累积概率阈值来工作。模型会选择那些累积概率达到此阈值的令牌集合，然后从这个集合中进行采样。较低的Top-P值会限制模型只考虑最有可能的令牌，从而产生更自信和事实性的响应；而较高的Top-P值则允许模型考虑更多可能性，从而产生更多样化的输出 14。通常建议只调整Temperature或Top-P其中一个，而不是同时调整，因为它们都影响输出的随机性 15。  
  * **Max Length (最大长度)**：此设置限制了AI模型生成响应的总令牌数量。它对于控制输出的冗长程度以及管理模型运行成本非常有用，可以防止生成过长或不相关的响应 14。  
  * **Stop Sequences (停止序列)**：停止序列是告知模型何时停止生成输出的特定字符串。通过设置这些序列，可以精确控制生成内容的长度和结构。例如，在生成电子邮件时，可以将“此致敬礼，”设置为停止序列，以确保模型在结束语前停止 14。  
  * **Frequency Penalty (频率惩罚)**：这个参数通过惩罚生成文本中重复出现的令牌来抑制重复。令牌在文本中出现的频率越高，模型再次使用它的可能性就越低 14。  
  * **Presence Penalty (存在惩罚)**：与频率惩罚类似，但存在惩罚对所有重复出现的令牌施加相同的固定惩罚，无论其出现频率如何。这个设置旨在防止模型重复短语 14。与Temperature和Top-P类似，通常建议只调整频率惩罚或存在惩罚其中一个 15。

#### **1.2. 基础提示词设计原则**

有效的提示词是与LLM成功交互的基石。以下是几个基础且至关重要的设计原则：

* **清晰与具体**: 编写提示词时，务必明确定义所需的响应，以避免AI模型的误解 1。这意味着要使用明确的指令、行动动词和量化要求。例如，与其说“总结小说”，不如明确指出“总结这本小说，我需要一个概要，而不是详细分析” 1。在需要特定格式时，例如“列出1990年代最受欢迎的电影，以表格形式呈现，并列出10部电影”，则需要明确指定数量和格式 1。  
* **提供上下文**: 在提示词中提供足够的背景信息和输出要求，有助于模型更好地理解任务并限定其响应的特定格式 1。模糊的提示词往往导致不相关或含糊的响应。例如，与其笼统地问“告诉我历史冲突”，不如具体说明“告诉我第二次世界大战的起因” 7。对于复杂或领域特定的语言，提供充足的上下文尤为重要，因为AI模型可能对此类信息不熟悉 1。  
* **平衡简洁与复杂**: 提示词的设计需要在简洁性和复杂性之间找到平衡。过于简单的提示词可能缺乏必要的上下文，而过于复杂的提示词则可能让AI感到困惑 1。目标是使用简单的语言，并适当减少提示词的长度，使其问题更容易被模型理解 1。  
* **迭代与实验**: 提示词工程是一个本质上的迭代过程 1。很少有提示词能在第一次尝试时就完美无缺。因此，持续实验不同的措辞、结构和参数，并测试AI的响应，是优化准确性和相关性的关键。通过反复尝试和调整，可以逐步完善提示词，使其生成更优质的输出 1。  
* **肯定性指令**: 在指示模型时，应采用肯定的指令，例如使用“做”（do）而非“不要”（don't）之类的否定性语言 7。例如，与其说“写一篇没有被动语态的文档”，不如说“写一篇使用主动语态的文档” 7。明确告知模型期望的行为，而非避免的行为，可以减少误解的可能性并提高输出质量 7。  
* **引入目标受众**: 在提示词中整合预期的受众，能够帮助模型调整输出的风格和内容，使其更符合目标群体的需求 19。例如，在撰写产品描述时，可以指定“面向关注可持续发展的年轻成年人” 17。  
* **使用分隔符**: 为了帮助模型更好地理解提示词中不同部分的区分，建议使用明确的分隔符，例如三引号（"""）、XML标签或换行符 19。这种“良好提示词卫生”的实践能够提高模型对指令的遵循度，从而产生更好的输出 19。  
* **角色扮演**: 为AI模型分配一个特定角色或视角，是一种强大的技术，可以引导其输出风格、语气和专业知识 18。例如，可以指示模型“你是一名专家级营销顾问”，这会促使模型以该角色的专业知识和沟通方式来响应 22。  
* **少量示例 (Few-Shot Prompting)**: 少量示例提示是一种通过在提示词中包含少量输入-输出示例来指导模型的技术 7。模型会从这些示例中学习所需的模式、风格和格式，并将其应用于新的输入。这种方法在需要特定风格或细节水平的专业主题上尤其有效 22。例如，提供几个已总结好的文本示例，可以帮助模型在后续总结中模仿相同的风格和深度 22。  
* **引导词**: 使用“一步一步思考”（think step by step）等引导词来鼓励思维链推理，这有助于模型在处理复杂任务时产生更结构化和高质量的输出 18。  
* **输出引子**: 通过在提示词的末尾提供期望输出的开头部分，可以引导模型生成特定格式的响应 19。这就像给模型一个“启动器”，帮助它沿着正确的方向继续生成。  
* **避免礼貌用语**: 在与LLM交互时，通常无需使用“请”、“谢谢”或“如果您不介意”等礼貌用语 19。直接切入重点，使用简洁明了的指令，可以提高效率并避免不必要的令牌消耗 19。  
* **重复关键词**: 在提示词中多次重复特定词语或短语，可以强调这些词语的重要性，从而引导模型更关注这些概念 19。

### **2\. 核心原则**

在掌握了基础概念和设计原则之后，深入理解零样本、少样本、思维链等核心提示词技术，以及如何有效管理上下文和利用角色扮演，对于提升提示词工程能力至关重要。

#### **2.1. 零样本、少样本与多样本提示**

这些是根据模型在提示词中接收到的示例数量来分类的提示词方法。

* 零样本提示 (Zero-Shot Prompting)  
  零样本提示是指在不提供任何先前示例或指导的情况下，直接向模型给出指令或问题 17。模型完全依赖其在预训练阶段获得的知识来解释和响应提示。这种方法适用于需要快速、通用响应的任务，例如进行头脑风暴、生成创意想法，或对文本进行总结和翻译等直接任务 17。  
* 少样本提示 (Few-Shot Prompting)  
  少样本提示通过在提示词中包含少量输入-输出示例来指导模型 7。模型会从这些明确的示例中识别模式和关系，并将其应用于生成对新输入的响应。这种方法在需要模型模仿特定风格、语气或细节水平的任务中特别有效，例如在处理专业主题或需要特定格式输出时 22。通过提供示例，模型能够更好地理解任务和期望的输出格式 24。然而，少样本提示的缺点在于它会增加提示词的长度，这可能导致更高的计算成本和延迟，并且存在模型从示例中学习到非预期模式的风险 24。  
* 多样本提示 (Multi-Shot Prompting)  
  多样本提示是少样本提示的扩展，它提供更多数量的示例，以期进一步提高模型对任务的理解和输出质量。当少量示例不足以完全捕获所需的复杂模式或多样性时，增加示例数量可以帮助模型更全面地学习。

#### **2.2. 思维链 (Chain-of-Thought, CoT) 提示**

思维链（CoT）提示是一种革命性的技术，它通过引导大型语言模型生成一系列中间推理步骤，显著提高了模型解决复杂问题的能力 26。这种方法的核心在于鼓励模型像人类一样，将复杂的难题分解为更小、更简单的逻辑步骤，并明确地展示这些步骤，从而提升其在算术、常识和符号推理任务上的性能 4。

* **定义与原理**: 传统的提示词通常直接寻求最终答案，而CoT则要求模型在给出答案之前，先阐明其思考过程。这种透明的逐步推理不仅有助于模型得出更准确的结论，还为人类提供了一个“窗口”，可以观察模型的行为并调试其推理路径中的错误 4。  
* **变体**: CoT提示已经发展出多种变体，以适应不同的应用场景和优化目标：  
  * **零样本CoT (Zero-Shot CoT)**: 这种方法通过在原始提示词中添加一个简单的通用短语，例如“一步一步思考”（"Let's think step by step"），来引导模型生成推理链，而无需提供任何具体的示例 18。尽管其简单有效，但有时可能导致推理错误或产生不准确的答案 26。  
  * **少样本CoT (Few-Shot CoT)**: 这是CoT的原始形式，它结合了少样本提示的理念，通过提供人工精心制作的示例来指导模型。每个示例都包含一个问题及其详细的推理链 4。这种方法比零样本CoT更可靠，因为它为模型提供了明确的学习模式，但缺点是需要耗费人工来创建这些示范 29。  
  * **自洽性 (Self-Consistency)**: 为了进一步提高复杂问题的推理性能，自洽性方法通过多次采样不同的推理路径，并选择其中最一致或多数投票的答案作为最终结果 18。这种方法增强了模型的鲁棒性，尤其是在问题可能存在多种解决途径时 29。  
  * **Auto-CoT**: 自动化思维链（Auto-CoT）旨在解决零样本CoT可能出现的错误和少样本CoT所需的大量手动工作。它通过自动生成多样化示例来改进CoT方法。其过程通常包括：首先，对问题数据集进行聚类，然后从每个聚类中选择一个或两个代表性问题，并使用零样本CoT为它们生成推理链。最后，将这些自动生成的推理序列插入到新问题的提示词中 26。尽管这种方法减少了人工干预，但其生成的多样性示例有时可能导致推理模式不一致或与实际问题不完全相关 26。

#### **2.3. 上下文管理与角色扮演**

有效利用模型的上下文能力是提示词工程中的关键技能。

* **上下文窗口的有效利用**: LLM的上下文窗口是其“记忆”的限制，它决定了模型在生成响应时能够考虑多少信息 13。为了确保模型能够充分利用所有相关信息，理解并管理这一限制至关重要。研究表明，模型在处理长输入时，对位于开头或结尾的信息表现最佳，而对中间部分信息的关注度可能下降 13。因此，在设计提示词时，应策略性地将最关键的信息放置在提示词的起始或结束部分，以最大化其被模型有效利用的可能性。  
* **系统消息与用户消息**: 在与LLM进行API交互时，通常会区分不同类型的消息角色，这些角色带有不同的权限和优先级。instructions API参数或developer消息角色提供了高层级的指令，定义了模型的行为、语气和总体目标，这些指令的优先级高于用户消息。user消息角色则包含来自终端用户的具体输入和配置，模型会根据developer消息中设定的规则来处理这些输入。模型自身生成的响应则归属于assistant角色 18。理解这些角色及其优先级，有助于构建更健壮和可控的AI应用。  
* **角色扮演**: 为AI模型分配一个特定角色或身份是一种强大的提示词技术，可以极大地引导其输出的风格、语气和专业知识 18。例如，通过指示模型“你是一名经验丰富的软件工程师”，模型在生成代码或提供技术建议时，会采用更专业、更符合该角色的语言和思维模式 22。这不仅提高了输出的相关性，也使得模型在特定领域内的表现更加出色。

### **3\. 高级技巧**

掌握了基础和核心原则后，可以进一步探索更复杂、更强大的提示词工程技巧，以应对更具挑战性的AI应用场景。

#### **3.1. 思维树 (Tree-of-Thought, ToT) 提示**

思维树（ToT）提示是一种比思维链（CoT）更通用的框架，它将LLM的推理过程建模为树状结构中的探索，而非简单的线性序列 32。这种方法允许模型并行生成和评估多个推理分支，主动识别、评估并修剪无用路径，从而在需要非平凡规划或搜索的复杂任务中显著提升问题解决能力 33。

* **定义与原理**: 在ToT框架中，“思想”（thoughts）被定义为连贯的语言序列，它们是解决问题过程中的中间步骤 37。与CoT的单一路线不同，ToT鼓励模型进行更深思熟虑的决策，通过考虑多个不同的推理路径，并自我评估选择以决定下一步行动，甚至在必要时进行前瞻或回溯，以做出全局性的选择 36。这使得ToT特别适合解决需要探索性和规划能力的复杂任务，例如24点游戏、创意写作和迷你填字游戏等 36。  
* **工作机制**: ToT的工作流程可以概括为以下几个步骤：  
  * **生成“思想”**: 从初始输入开始，模型会生成多个潜在的“思想”或解决方案。每个“思想”都是解决问题的一个可管理的小步骤 34。例如，在解决数独谜题时，一个“思想”可能涉及尝试不同的数字放置 34。  
  * **评估“思想”**: 在每一步，模型会评估这些生成的“思想”，判断哪些是值得深入探索的（有前景的），哪些应该被放弃（无用或导致矛盾） 35。评估可以基于标量值（如评分）或分类（如“确定”、“可能”、“不可能”） 34。  
  * **扩展有前景的思想**: 那些被评估为有前景的“思想”会进一步分支，生成新的节点，从而扩展探索空间 35。  
  * **搜索最佳解决方案**: 模型会利用搜索算法（如广度优先搜索BFS或深度优先搜索DFS）来导航这个思想树。BFS会探索每个层级的所有可能分支，适用于需要最短路径或浅层解决方案的问题；而DFS则会深入探索一个分支，适用于需要详细探索每个选项的问题 34。  
* **优势**: ToT框架具有多项显著优势。它允许模型同时探索多条推理路径，这极大地增加了在复杂或模糊问题中找到最佳解决方案的可能性 35。通过在早期阶段放弃较弱或无用的想法，ToT能够有效防止计算资源的浪费，并专注于最有前景的分支 35。此外，ToT的适应性强，不局限于特定领域，无论是解决复杂的数学问题、创作富有想象力的故事，还是优化物流计划，该框架都能根据每个场景的独特挑战进行调整 35。

#### **3.2. 检索增强生成 (Retrieval-Augmented Generation, RAG)**

检索增强生成（RAG）是一种将传统信息检索系统（如搜索引擎和数据库）与生成式大型语言模型（LLM）能力相结合的AI框架 21。这种方法通过从外部知识库检索相关文档块来增强LLM的生成能力，从而有效减少幻觉（即模型生成不准确或虚假信息）、提供最新知识并提高透明度 21。

* **定义与原理**: 纯粹的LLM在训练数据之外可能存在知识限制，容易产生“幻觉”或提供过时信息 38。RAG通过引入一个检索模块来克服这些挑战，该模块在生成响应之前先从外部数据源（如企业内部文档、实时网页数据）中获取相关信息。这些检索到的信息随后被作为额外的上下文提供给LLM，从而使模型能够生成更准确、更具事实依据的响应 38。  
* **架构演进**: RAG框架经历了从简单到复杂的演变，以不断提升其性能和适应性：  
  * **朴素RAG (Naive RAG)**: 这是RAG的最基本形式，遵循“检索-读取”框架。其过程包括：  
    1. **索引**: 原始数据（如PDF、HTML、Word文档）被清洗并转换为纯文本，然后分割成适合LLM上下文限制的“块”。这些文本块通过嵌入模型转换为向量表示，并存储在向量数据库中，以便进行高效的语义相似性搜索 38。  
    2. **检索**: 当接收到用户查询时，查询本身也会被相同的嵌入模型转换为向量表示。系统随后计算查询向量与索引语料库中所有文本块向量的相似度，并检索出最相关的K个块。这些检索到的块作为扩展上下文被整合到LLM的提示词中 38。  
    3. **生成**: 用户查询和检索到的文档被结合成一个连贯的提示词。LLM随后根据这些信息生成响应，其输出可以基于其内在知识，也可以被限制在检索到的信息范围内。对于持续对话，对话历史也可以被整合到提示词中 38。  
  * **高级RAG (Advanced RAG)**: 为了解决朴素RAG的局限性，高级RAG引入了改进策略，主要关注通过预检索和后检索阶段来提升检索质量。  
    1. **预检索过程**: 优化索引结构和原始查询。索引优化包括提升数据粒度、精炼索引结构、添加元数据、对齐优化和混合检索。查询优化旨在使用户的问题更清晰、更适合检索，方法包括查询重写、查询转换和查询扩展 38。  
    2. **后检索过程**: 在检索到相关上下文后，此阶段侧重于与查询的有效整合。关键方法包括对文本块进行重排序（将最相关的内容放置在提示词的开头或结尾）和上下文压缩（以减轻信息过载）38。  
  * **模块化RAG (Modular RAG)**: 模块化RAG代表了RAG架构的进一步发展，提供了增强的适应性和通用性。它引入了更多样化的策略来改进组件，例如添加专门的搜索模块和通过微调来改进检索器。模块化RAG支持其组件的顺序处理和集成端到端训练，允许更灵活的模块替换或重新配置 5。  
    1. **新模块**: 该框架引入了如搜索模块（用于跨多种数据源直接搜索）、RAG-Fusion（用于多查询策略）、内存模块（利用LLM的记忆来指导检索）、路由（导航多样化数据源）、预测模块（通过LLM直接生成上下文以减少冗余）和任务适配器模块（根据各种下游任务定制RAG）等专用组件 38。  
    2. **新模式**: 模块化RAG允许更灵活的模块排列和交互，超越了固定的“检索”和“读取”机制。示例包括重写-检索-读取（通过LLM精炼查询）、生成-读取（用LLM生成的内容替代传统检索）和背诵-读取（从模型权重中检索）。它还集成了混合检索策略和自适应检索技术，如FLARE和Self-RAG，这些技术使RAG系统能够自主决定何时需要检索外部知识 5。  
* **优势**: RAG具有多项关键优势。它能够让LLM访问最新信息，克服其预训练数据固有的过时问题 39。通过引用外部知识，RAG有效地减少了LLM生成不准确或无事实依据内容（幻觉）的问题，从而显著提高了生成内容的准确性和可信度 21。此外，RAG支持集成领域特定信息，使LLM更适用于其初始训练数据可能不足的专业任务 21。它还提供了更透明和可追溯的推理过程，因为生成的答案可以追溯到具体的检索文档 21。从成本效益角度看，RAG通常比频繁地重新训练LLM更经济，因为它利用外部知识库而无需对每次更新进行大量模型微调 38。对于长文档问答，RAG的分块检索和按需输入可以显著提高操作效率 38。最后，RAG向高级和模块化范式的演进带来了更大的灵活性、检索质量优化、新功能模块的集成以及自适应检索过程，以适应多样化的场景和复杂问题 5。

#### **3.3. 自动化提示词优化 (Automatic Prompt Optimization, APO)**

自动化提示词优化（APO）技术利用自动化流程来改进大型语言模型（LLM）在各种自然语言处理（NLP）任务中的性能 5。随着模型、任务和最佳实践的快速演进，手动进行提示词工程变得越来越具有挑战性。APO旨在通过自动化提示词的改进过程来缓解这一问题 8。

* **定义与原理**: APO的核心在于，给定一个任务模型和一个初始提示词，APO系统的目标是根据特定的评估指标和验证数据集，找到表现最佳的提示词模板 8。这个过程通常是一个优化问题，旨在最大化模型在评估集上的预期性能 8。APO的吸引力在于它通常不需要访问LLM的内部参数，能够系统地搜索提示词解决方案空间，并且改进后的提示词仍然保持人类可解释性 8。  
* **核心技术**: APO涵盖了一系列技术，共同构成了其优化框架：  
  * **种子提示生成**: 优化过程从生成一组初始提示词开始。这可以是从人类手动创建的指令开始，这些指令通常作为进一步优化的可解释基线。另一种方法是利用LLM自身来推断人类可读的提示词，从少量示例中学习并生成初始指令 8。  
  * **推理评估与反馈**: 这一步对于在每次迭代中识别有前景的提示词候选至关重要。评估可以通过多种方式进行：  
    * **数值分数反馈**: 使用任务特定的准确性指标（如分类或问答的精确度、代码任务的执行准确性），或更灵活的指标（如BLEU、ROUGE、BERTScore）来评估文本生成任务 8。此外，还可以使用学习到的奖励模型来提供更细致的评估，预测提示词是否能引出正确答案 8。一些方法还利用熵分数或输出的负对数似然进行评估，但这通常需要访问模型的内部概率分布 8。  
    * **LLM反馈**: 另一种流行的方法是让LLM本身生成文本反馈来评估提示词-响应对。这种方法通用性强，因为它只需要自然语言指令，但会增加额外的推理成本 8。LLM可以被指示改进单个提示词候选，或生成多个候选并提供文本“梯度”来指导优化 8。  
    * **人类反馈**: 一些方法将人类反馈整合到提示词的构建或优化过程中，无论是编译时还是推理时。这可能涉及交互式提问以推断人类偏好，或训练较小的LLM根据用户偏好优化提示词 8。  
  * **候选提示词生成**: 这一步涉及创建新的提示词变体，以期提高性能。常用的方法包括：  
    * **启发式编辑**: 对中间提示词候选进行离散编辑，例如使用蒙特卡洛采样、遗传算法（通过变异和交叉操作）、词/短语级别编辑（替换同义词、添加、删除、改写、交换）或词汇剪枝 8。  
    * **通过辅助训练神经网络进行编辑**: 使用一个辅助的神经网络来编辑初始提示词，例如通过强化学习、微调LLM或生成对抗网络（GAN）来优化提示词 8。  
    * **元提示设计**: 探索元提示的搜索空间，包括自然语言中的优化问题描述和先前生成的解决方案，以指导提示词的精炼 8。  
    * **基于覆盖的方法**: 旨在“覆盖”整个问题空间，无论是通过单个扩展提示词还是使用多个提示词的集合 8。  
    * **程序合成**: 将LLM管道转换为结构化、模块化的组件，可以系统地优化和组合，迭代地精炼每个模块的指令和示例 8。  
  * **过滤与保留**: 这一步选择最有前景的提示词候选进行进一步优化。最简单的方法是保留在小批量数据上表现最佳的Top-K个候选 8。更高级的方法可能将候选提示词选择视为一个多臂老虎机问题，平衡探索和利用以在固定计算预算内识别合适的提示词 8。  
  * **迭代深度**: 确定优化过程将运行多少步。大多数方法会运行预设的步数，而有些方法则在连续迭代的性能提升低于某个阈值或达到特定性能目标时停止搜索 8。  
* **优势**: APO技术具有显著的优势。它们无需访问LLM的内部参数，这使得它们可以应用于各种黑盒模型 8。它们能够系统地搜索提示词解决方案空间，而非依赖人工的反复试错 8。此外，APO生成的提示词改进通常保持人类可解释性，这有助于理解模型行为并进行进一步的调整 8。

#### **3.4. 自我修正 (Self-Refinement) 与提示词链 (Prompt Chaining)**

这些高级技巧使LLM能够处理更复杂的任务，并生成更连贯、高质量的输出，模拟人类解决问题和创作的过程。

* 自我修正 (Self-Refinement)  
  自我修正是一种模仿人类写作中批判和完善过程的方法，旨在通过迭代反馈和修正来改进LLM的初始输出 5。其核心在于，LLM不仅能生成内容，还能对自己的生成内容进行评估并提供改进建议，然后根据这些反馈进行修正。  
  * **过程**: 自我修正的过程通常包括三个顺序步骤：  
    1. **初始生成**: LLM首先根据提示词生成一个初步的响应或草稿 47。  
    2. **自我批判**: 随后，同一个LLM会对其刚刚生成的输出提供批判性反馈和有益的建议 47。  
    3. **迭代修正**: 最后，LLM利用这些自我提供的反馈来修正其初始输出，这个过程可以循环进行，直到达到预期的质量或满足特定条件 47。  
  * **优势**: 这种方法的显著优势在于它不需要任何额外的监督训练数据、额外的模型训练或强化学习。它仅使用单个LLM作为内容的生成器、修正器和反馈提供者 48。这使得自我修正成为一种高效且灵活的优化策略，适用于各种文本生成任务，例如摘要、代码生成和文本到SQL转换等 46。  
* 提示词链 (Prompt Chaining)  
  提示词链是一种将复杂任务分解为一系列离散的、顺序的提示词的技术 3。在这种模式下，前一个步骤的输出会作为下一个步骤的输入，从而构建一个多步骤的工作流。  
  * **优势**: 提示词链的主要优势在于它允许模型一次专注于一个子问题，从而避免因任务过于复杂而导致的性能下降或混淆 47。例如，在撰写一份报告时，可以先提示模型生成大纲，然后根据大纲生成执行摘要，再逐步填充各个部分 7。这种迭代和分解的方法使得模型能够更系统地处理复杂问题。研究表明，提示词链方法在文本摘要等任务中，相较于将所有指令集成在一个单一提示词中的“单步提示词”方法，能够产生更优的结果 47。这表明，通过明确地将任务分解为逻辑步骤，可以引导模型进行更深入和高质量的思考。

#### **3.5. 其他高级技巧**

除了上述核心高级技巧，还有一些值得关注的方法可以进一步提升提示词工程的效果：

* **Maieutic Prompting (产婆式提示)**: 这种方法类似于思维树，它要求模型在给出答案后，进一步解释其答案的各个部分。如果模型在解释过程中出现不一致或矛盾，相关的解释树会被剪除或修正 1。这种方法有助于提高模型在复杂常识推理任务上的表现，因为它鼓励模型进行更深层次的自我验证。  
* **Meta-prompting (元提示)**: 元提示可以指使用更高层次的提示词来控制多个下游指令的语气、结构或安全性 5。它也可以指利用LLM自身的能力来分析和改进现有的提示词 44。例如，可以提示模型“创建一个能帮助你解释气候变化的提示词”，让模型生成一个更有效的提示词来完成任务 25。这种自我引导的能力可以显著提高输出的质量和相关性。  
* **Output Primers (输出引子)**: 这种技巧通过在提示词的末尾提供期望输出的开头部分来引导模型生成特定格式的内容 19。例如，如果期望一个列表，可以在提示词末尾写上“1.”，模型就会倾向于继续生成列表项。  
* **Dynamic Prompting (动态提示词)**: 动态提示词允许程序在提示词中随机选择列表中的令牌，从而在单个提示词中生成最大程度的变化和自发性输出 53。例如，  
  {red|green|blue} suit 可以生成“红色套装”、“绿色套装”或“蓝色套装”。这种方法适用于需要多样化和创意性输出的场景。  
* **Prompt Weighting (提示词加权)**: 提示词加权是指通过强调或降低提示词中某些术语的重要性，使特定特征、风格或效果在最终输出中更突出或不突出 53。例如，  
  (Smiling:0.5) 与 (Smiling:1.2) 可能会导致从轻微微笑到灿烂笑容的不同结果，甚至可能改变图片的构图焦点。这种方法在图像生成中尤其常见，用于精细控制生成内容的细节。  
* **Multimodal Prompting (多模态提示词)**: 随着AI模型能够处理文本、图像、音频甚至视频等多种输入，多模态提示词变得越来越重要 5。它涉及整合不同模态的输入来生成响应，例如“分析这份合同和相关图表”或“根据会议音频和演示文稿生成报告” 5。未来的提示词工程将更多地是设计上下文生态系统，使AI系统能够摄取、对齐和响应多样化的信息流，形成连贯、统一的工作流 5。

### **4\. 特定领域应用**

提示词工程的技巧并非一概而论，针对不同的AI模型和应用领域，需要采取特定的策略以达到最佳效果。

#### **4.1. 文本生成 (Text Generation)**

文本生成是LLM最常见的应用之一，包括摘要、翻译和创意写作等。

* 摘要:  
  在进行文本摘要时，明确目标至关重要。例如，可以指定“突出关键论点”、“聚焦伦理影响”或“提炼成三个要点” 22。同时，提供简洁的指南，如字数限制（“在50字内总结”）或偏好格式（如项目符号列表、短段落或编号列表）22。指定目标受众（如学生、高管或普通读者）有助于模型调整摘要的复杂度和风格 22。此外，提供风格线索，如“正式”或“会话式”的语气，也能确保输出符合预期 22。  
* 翻译:  
  在进行翻译任务时，明确源语言和目标语言是基础。提供足够的上下文和背景信息，并使用分隔符清晰地标记待翻译的文本，可以帮助模型更准确地理解和翻译 54。  
* 创意写作:  
  对于创意写作任务，设定具体的创作目标和目标受众非常重要 23。提供丰富的上下文和背景信息，例如故事的设定、人物性格或期望的情节发展，能够引导模型生成更符合预期的创意内容 23。利用角色扮演技巧，让AI扮演特定作家或风格的专家，可以帮助模型模仿特定的写作风格 23。此外，由于创意写作的开放性，迭代和实验是必不可少的，需要不断调整提示词以探索不同的创意方向和完善输出 23。

#### **4.2. 代码生成与调试 (Code Generation & Debugging)**

对于具备编程经验的学习者而言，利用提示词工程进行代码生成和调试是提升开发效率的关键。

* **代码生成**:  
  * **先泛后精**: 在请求代码生成时，首先提供对目标或场景的宽泛描述，然后逐步列出所有具体的实现要求和约束。例如，先提出“编写一个判断数字是否为素数的函数”，再细化为“该函数应接受一个整数，如果为素数则返回true，如果输入不是正整数则报错” 50。  
  * **提供示例**: 包含示例是提高代码生成质量的有效方法。可以提供示例输入数据、预期输出，甚至示例实现代码 50。单元测试也可以作为强有力的示例；可以先让模型编写单元测试，然后再要求它根据这些测试生成函数 50。  
  * **分解复杂任务**: 对于大型或复杂的编码任务，应将其分解为多个更小、更简单的子任务。例如，与其一次性要求生成整个文字搜索谜题，不如分步要求：先生成一个10x10的字母网格，再生成一个在网格中查找单词的函数，以此类推 50。这种增量方法允许在每个组件构建完成后进行审查和修正 51。  
  * **避免歧义**: 使用精确的语言，避免模糊的术语。例如，不要问“这个做什么？”，而是明确指出“createUser函数做什么？”或“你上次响应中的代码做什么？” 50。如果使用不常见的库，应描述其功能；如果需要使用特定库，则应明确指定或包含导入语句 50。  
  * **指明相关代码**: 在集成开发环境（IDE）中使用代码助手时，应打开所有相关文件并关闭不相关文件，因为模型会利用打开的文件来理解请求的上下文 50。在聊天界面中，可以高亮显示需要模型参考的代码块，或使用  
    @workspace或@project等关键词手动提供上下文 50。  
  * **迭代与实验**: 如果初始提示词未能产生所需结果，应反复调整和优化提示词。可以删除不理想的建议并重新开始，或者在现有建议的基础上请求修改 50。  
  * **保持历史相关性**: 聊天历史为模型提供了上下文。为了确保上下文的有效性，应为每个新任务启动新的对话线程，并删除不再相关或未产生预期结果的请求 50。  
  * **遵循良好编码实践**: 确保现有的代码库清晰、遵循最佳实践（如一致的代码风格、描述性命名、模块化组件和单元测试），这将有助于模型生成更好的建议和解释 50。  
* 代码调试:  
  在进行代码调试时，向AI提供详细的错误信息至关重要，包括完整的错误消息、堆栈跟踪和任何相关日志。同时，提供出错的代码片段、预期的代码行为以及已尝试过的解决方案，能够帮助AI更快地定位问题并提供有效的修复建议。

#### **4.3. 图像生成与编辑 (Image Generation & Editing)**

图像生成和编辑是生成式AI的另一个重要应用领域，提示词工程同样扮演着核心角色。

* **文本到图像生成**:  
  * **从简单到复杂**: 开始时使用简单的提示词，然后逐步添加更多元素和上下文，以实现更精细的结果 54。这种迭代方法可以避免一开始就使提示词设计过程过于复杂 54。  
  * **指令明确**: 使用“绘制”、“生成”等明确的命令来指示模型，并尽可能详细地描述期望的风格、主题、构图、光照、颜色和情绪等细节 53。例如，可以指定相机和镜头类型以获得特定的摄影风格 54。  
  * **提供示例**: 通过提供参考图像或描述性示例，可以有效指导模型生成期望的风格、格式或细节水平的图像 54。  
  * **提示词结构**: 某些图像生成模型（如Stable Diffusion）对提示词中令牌的顺序敏感。研究表明，推荐的提示词结构通常是：主体、细节、背景、风格、光照 53。遵循这种结构有助于模型更好地理解和生成所需图像。  
  * **动态提示词与加权**: 动态提示词允许在提示词中随机选择元素，从而生成多样化的输出 53。提示词加权则用于强调或降低提示词中某些术语的重要性，使特定特征、风格或效果在最终输出中更突出或不突出 53。  
  * **指定长宽比**: 始终在提示词中明确指定期望的图像长宽比，即使提供参考图像，模型在未明确指令时也可能默认生成1:1的比例 54。  
* **图像编辑**:  
  * **上下文特异性**: 在进行图像编辑时，添加场景的上下文可以帮助模型更好地理解并生成适当的输出。这种特异性有助于模型专注于相关方面，避免产生无关细节 56。例如，在为产品目录描述图像时，提示词应反映出“为户外徒步产品目录描述图像，侧重于热情和专业性” 56。  
  * **任务导向**: 给予模型一个具体的任务，能够引导模型从该视角生成输出，从而提高准确性和相关性 56。  
  * **处理拒绝**: 当模型指示无法执行某项任务时，精炼提示词通常是有效的解决方案 56。  
  * **图片置于文本前**: 如果是单图片提示，建议将图片放置在文本内容之前 56。  
  * **先描述图片再任务**: 可以要求模型首先详细描述图片内容，然后根据其描述完成特定的编辑任务 56。  
  * **定义输出格式**: 明确指定所需的输出格式，例如Markdown、JSON或HTML，可以帮助模型生成结构化的编辑结果 56。

### **5\. 评估与优化**

提示词工程是一个持续迭代和优化的过程。为了确保提示词能够持续高效、稳定且富有创造力地工作，需要系统地评估其性能并实施优化策略。

#### **5.1. 提示词评估方法**

评估LLM的输出是提示词工程闭环中不可或缺的一环，它包括人工和自动化两种主要方法。

* 人工评估 (Human-in-the-Loop, HITL):  
  人工评估涉及真实的人类审查员对模型输出进行评分和反馈 57。这种方法对于评估主观、细致和高风险任务的输出质量至关重要，例如创意写作、情感分析或法律文本的准确性。尽管人工评估成本较高且速度较慢，但它在捕捉细微错误、处理边缘情况以及评估模型推理质量方面，仍然是最可靠的方法 57。许多大型科技公司，如Meta，都在人工审查服务上投入巨资，这突显了HITL在模型开发中的关键作用 57。人工监督还能防止模型因反复训练AI生成内容而导致的性能退化，确保质量始终与现实世界保持一致 57。  
* LLM作为评估者 (LLM-as-a-judge):  
  这种方法利用一个AI模型来评估另一个AI模型的输出，从而实现快速、大规模的数据处理 57。LLM作为评估者在效率上具有显著优势，可以持续监控性能而无需人工审查每一个输出 57。然而，这种方法存在潜在的偏见，AI评估者可能偏爱某些响应，或遗漏人类可感知的细微上下文 57。此外，还可能出现“回音室效应”，即AI评估者倾向于认可与自身编程相似的响应，从而可能忽略更具创造性的答案 57。为了验证AI评估者的准确性，通常需要让人类审查员评估一部分模型响应，然后让AI评估者处理相同的响应。如果两者的一致性很高（通常高于85-90%），则AI评估者可以投入自动使用 57。LLM作为评估者在处理主观评估、涉及多重因素（如语气或实用性）的评估、新颖或不熟悉的内容，或当出现新型故障时，仍面临挑战，此时人类判断仍然是必要的 57。  
* 结合人工与LLM评估:  
  最可靠的评估方法是结合多层策略，充分利用人工评估的可靠性和LLM作为评估者的效率 57。通常，AI评估者会处理大部分初始评估工作。对于复杂、LLM评估者之间意见不一致，或者人类专家认为尽管AI批准但输出质量较差的情况，则会转交由人工审查 57。这种结合策略能够在保持高效率的同时，确保AI系统的可靠性和高质量输出 57。

#### **5.2. 常用评估指标**

根据任务类型，可以使用不同的指标来量化LLM提示词的性能。

* **文本生成**:  
  * **Perplexity (困惑度)**: 衡量模型预测文本的好坏。分数越低表示模型预测能力越强，性能越好。它计算样本平均对数似然的指数 57。然而，困惑度并不直接反映文本的质量或连贯性，且可能受到分词方式的影响 57。  
  * **BLEU Score**: 最初用于机器翻译，现在也用于评估文本生成。它通过比较模型输出与参考文本的N-gram重叠度来衡量相似性。分数范围从0到1，越高表示匹配度越好。但BLEU可能无法准确评估创意或多样化的文本输出 57。  
  * **ROUGE**: 适用于摘要评估。它衡量模型生成内容与参考摘要的N-gram、序列和词对的重叠度 57。  
  * **F1 Score**: 用于分类和问答任务。它平衡了精确率（模型响应的相关性）和召回率（相关响应的完整性），范围从0到1，1表示完美准确性 57。  
  * **METEOR**: 考虑精确匹配、同义词和释义，旨在更好地与人类判断保持一致 57。  
  * **BERTScore**: 通过比较BERT等模型生成的上下文嵌入的相似性来评估文本，更侧重于语义相似性而非精确的词匹配 57。  
  * **Levenshtein distance (编辑距离)**: 衡量将一个字符串转换为另一个所需的最小单字符编辑次数。它对于评估文本相似度、拼写校正和OCR后处理很有价值，但无法考虑语义相似性，因此最好与其他指标结合使用 57。  
  * **答案相关性 (Answer Relevancy)**: 评估检索增强生成（RAG）生成器输出的答案是否简洁且与输入相关。通过计算LLM输出中与输入相关的句子比例来衡量 58。  
  * **上下文精确度 (Contextual Precision)**: 评估RAG管道检索器的质量。高上下文精确度分数意味着检索到的相关节点排名高于不相关节点，这很重要，因为LLM会更重视检索上下文前面部分的信息 58。  
  * **上下文召回率 (Contextual Recall)**: 评估预期输出或真实数据中可归因于检索上下文的句子比例。分数越高表示检索到的信息与预期输出之间的一致性越高 58。  
  * **上下文相关性 (Contextual Relevancy)**: 检索上下文中与给定输入相关的句子比例 58。  
* 代码生成:  
  对于代码生成任务，评估指标通常包括代码的编译成功率和测试通过率 57。这些指标直接反映了生成代码的功能正确性。  
* 通用指标:  
  除了特定任务的指标，还有一些通用指标适用于所有LLM应用，例如任务特定指标（如对话系统中的用户参与度、任务完成率）和效率指标（如响应速度、内存使用量和能源消耗）57。随着模型规模的增长，效率指标的重要性日益凸显 57。

#### **5.3. 提示词优化策略**

提示词优化是一个持续改进的过程，旨在确保提示词能够持续高效、稳定且富有创造力地工作。

* 迭代优化:  
  提示词优化本质上是一个迭代过程，涉及对初始提示词进行多次精炼，以达到高质量和可靠的输出 43。这个过程的核心是持续的测试、输出评估和改进。它包括使用人工判断或自动化指标评估输出，根据清晰度、结构、特异性或长度进行调整，并在具有代表性的数据集上进行测试 43。一个常见的误区是跳过迭代，因为不测试变体或不比较输出会错失性能提升的机会 43。  
* A/B测试:  
  A/B测试是一种系统地比较两个或多个提示词变体的方法，以确定哪个能带来更优响应 31。其核心在于将目标受众平均分成几组，每组只接触一个提示词变体，然后收集和分析数据（如点击率、参与度、转化率）来评估效果 59。  
  * **关键组成**: A/B测试包括假设制定（预测提示词变化的影响）、变体创建（设计对照组和变体组）、受众细分（确保无偏性）以及数据收集与分析 59。  
  * **最佳实践**: 实施A/B测试时，需要明确目标，确保细分一致性，达到统计显著性，采用迭代测试方法，并始终以用户为中心 59。  
* 提示词版本控制:  
  像软件版本控制一样管理和跟踪提示词的变化，对于确保质量、简化故障排除和改善团队协作至关重要 24。  
  * **最佳实践**: 采用语义版本控制（如v1.0.0到v1.1.0），保持提示词清晰（包含精确指令、示例和限制），文档化所有变更（记录更改原因、目标和性能指标），并准备回滚机制（如功能标志和检查点，以便在出现问题时迅速恢复到稳定版本）61。  
  * **核心要素**: 良好的提示词版本控制应包含详细的版本历史、回滚到先前版本的能力、在部署前对提示词进行测试、管理不同提示词变体以进行A/B测试，以及跟踪不同环境中运行的提示词版本 62。  
* 自动化工具:  
  为了简化提示词的评估和管理，可以利用各种自动化工具。例如，LangSmith和Langfuse专门用于日志记录和提示词实验与评估 63。Mirascope是一个轻量级且用户友好的LLM工具包 63。Haystack擅长构建提示词管道，而Agenta则支持快速协作的LLM应用开发 63。LangChain提供可扩展和可定制的LLM应用框架，而Lilypad则专注于业务用户和软件开发者的协作式提示词工程 63。这些工具能够帮助开发者更高效地管理、测试和优化提示词，从而提升AI应用的整体性能和可靠性。

**第二部分 \- 精选学习资料**

为了系统性地掌握AI提示词工程，以下推荐一系列高质量、权威的学习资料，涵盖理论基础、官方指南、在线课程和实战项目。

### **1\. 必读论文**

这些论文是提示词工程领域的奠基性或具有里程碑意义的研究，深入理解它们有助于掌握该领域的核心思想和前沿发展。

* **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022\)** 4:

  这篇论文是思维链（CoT）提示词的开创性工作，它展示了通过引入一系列中间推理步骤，能够显著提高大型语言模型在复杂推理任务上的能力。它揭示了CoT如何让LLM模仿人类的逐步思考过程，从而在算术、常识和符号推理等任务中取得惊人的性能提升。  
* **Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022\)** 29:

  这篇论文引入了零样本思维链（Zero-Shot CoT）的概念，表明大型语言模型在仅通过一个简单提示（如“一步一步思考”）的情况下，无需任何特定示例，就能够展现出零样本推理能力。这为CoT的广泛应用打开了新的可能性。  
* **Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023\)** 36:

  该论文提出了思维树（ToT）框架，将CoT泛化为树状探索结构。ToT允许LLM通过并行生成和评估多个推理分支，进行深思熟虑的决策和规划。这使得LLM能够解决需要非平凡规划或搜索的复杂问题，如24点游戏和创意写作。  
* **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020\)** 41:

  这篇论文首次提出了检索增强生成（RAG）架构。RAG通过结合外部知识检索来增强LLM的事实准确性，并有效减少幻觉。它展示了如何将LLM的生成能力与外部数据库的最新、可靠信息相结合，从而在知识密集型NLP任务中取得显著改进。  
* **Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023\)** 47:

  该论文提出了一种自我修正的方法，LLM能够通过迭代反馈和修正来改进自身的初始输出，从而模仿人类写作的批判和完善过程。这种方法无需额外的监督训练数据，仅使用单个LLM作为生成器、修正器和反馈提供者。  
* **A Systematic Survey of Automatic Prompt Optimization Techniques (Ramnath et al., 2025\)** 8:

  这篇综述论文对自动化提示词优化（APO）技术进行了全面梳理。它提供了APO的正式定义、一个统一的五部分框架，并系统地对现有相关工作进行了分类，为理解和进一步研究自动化提示词改进提供了宝贵的参考。

### **2\. 官方文档或指南**

各大AI平台提供的官方文档是学习提示词工程的最佳起点，它们通常包含模型特定的最佳实践和技术。

* **OpenAI Prompt Engineering Guide** 20:

  OpenAI的官方指南提供了编写清晰指令、选择不同模型（如推理模型与GPT模型）、理解消息角色（instructions参数、developer、user、assistant消息）、利用可复用提示词、少样本学习以及集成检索增强生成（RAG）等策略。它还强调了上下文窗口的规划和利用Markdown与XML进行消息格式化的重要性。  
* **Google AI Prompt Design Strategies** 16:

  Google AI的指南涵盖了清晰具体的指令、不同输入类型（问题、任务、实体、完成）、约束设置、响应格式指定、零样本与少样本提示词的运用、上下文添加、提示词分解为组件、模型参数（如Max output tokens, Temperature, topK, topP）的实验，以及提示词迭代策略等。  
* **Hugging Face Prompt Engineering Guide** 24:

  Hugging Face的指南提供了针对LLM性能提升的关键提示词工程原则和技术。内容包括模型选择（基础模型与指令调优模型）、迭代优化、指令放置、清晰分离指令与文本、具体描述任务和输出、使用肯定性指令、输出引导、少样本提示词和思维链（CoT）等最佳实践。  
* **AWS What is Prompt Engineering?** 1:

  AWS的概述性文章解释了提示词工程的定义、其带来的益处（如增强开发者控制、改善用户体验、提高灵活性和可扩展性、降低成本和提升效率），并分享了编写有效提示词的最佳实践，如使用明确的提示词、提供足够的上下文、平衡信息与输出，以及持续实验和迭代。  
* **GitHub Copilot Prompt Engineering** 50:

  这份指南专门针对代码生成场景，提供了如何从GitHub Copilot Chat获取最佳结果的策略。建议包括：先泛后精、提供示例（包括单元测试）、分解复杂任务、避免歧义、指明相关代码、迭代实验、保持历史相关性，以及遵循良好的编码实践。

### **3\. 在线课程或教程**

在线课程提供了结构化的学习路径和实践机会，是系统学习提示词工程的有效途径。

* **Coursera**:  
  * **Generative AI: Prompt Engineering Basics (IBM)** 65: 本课程涵盖提示词的核心元素、有效编写方法，以及各种提示词工程技术，包括零样本、少样本、思维链（CoT）和思维树（ToT）。它还涉及多模态提示词和评估方法，并提供动手实验。  
  * **Google Prompting Essentials (Google)** 52: 该课程由Google专家设计，教授5步提示词框架、4种关键迭代方法、文本到文本和文本到图像提示词、多模态提示词、少样本提示词和提示词链等核心技术。  
  * **ChatGPT Prompt Engineering for Developers (DeepLearning.AI)** 65: 这门课程专注于为开发者设计，深入探讨了ChatGPT的提示词工程模式，并提供了API使用的实践指导。  
  * **Prompt Engineering for ChatGPT (Vanderbilt University)** 65: 课程内容涵盖了如何利用提示词工程提升生产力、激发创意、解决问题和进行应用开发。  
* Udemy:  
  Udemy平台上有众多关于提示词工程的课程，从基础入门到高级技巧，内容涵盖ChatGPT、AI内容生成、DALL·E、Midjourney、LangChain等热门工具和框架 67。学习者可以根据自己的需求和水平选择合适的课程。  
* Learn Prompting:  
  Learn Prompting提供免费的“ChatGPT for Everyone”课程 68，非常适合初学者。课程内容包括ChatGPT的基础知识、账户设置、有效提示词的创建、常见用例以及AI安全与局限性。该课程由行业专家授课，提供了实践性强且易于理解的入门指导。  
* Real Python \- Practical Prompt Engineering:  
  对于Python开发者，Real Python的“Practical Prompt Engineering”教程 69 通过一个实际项目，指导学习者应用多种提示词工程技术，如零样本、少样本、分隔符、分步指令、角色提示和思维链等，帮助学习者在实践中掌握这些技巧。

### **4\. 实战项目或挑战**

通过实际项目和挑战来巩固理论知识，是提升提示词工程技能的有效途径，特别是对于有编程背景的学习者。

* GitHub Repositories:  
  GitHub上汇集了大量与提示词工程相关的开源项目和教程，提供了丰富的实践案例。  
  * **NirDiamant/Prompt\_Engineering** 70: 这是一个综合性的教程和实现集合，涵盖了从基础概念到高级策略的提示词工程技术。包括零样本、少样本、思维链（CoT）、自洽性、角色扮演、任务分解、提示词链、优化、歧义处理、长度管理、负面提示词、格式化、特定任务提示词、多语言提示词和安全等。每个教程都包含详细的实现指南和代码示例。  
  * **microsoft/generative-ai-for-beginners** 72: 微软为初学者提供的生成式AI资源，包含Jupyter Notebooks，适合通过实践学习。  
  * **mlflow/mlflow** 72: 用于构建和部署智能AI/LLM应用的平台，提供端到端跟踪、可观测性和评估功能，对于理解LLM运维（LLMOps）和评估提示词性能很有帮助。  
  * **awesome-chatgpt-zh** 72: 一个中文ChatGPT相关资源的精选集合，可以找到更多中文社区的实践案例和讨论。  
  * **awesome-gpt4o-images** 72: 专门收集了使用GPT-4o和gpt-image-1生成图像和提示词的精选案例，对于图像生成领域的提示词工程有很好的参考价值。  
* Kaggle Datasets & Competitions:  
  Kaggle平台提供了大量数据集和机器学习竞赛，其中一些与提示词工程密切相关。  
  * **Prompt Engineering Transformation Dataset** 73: 这个数据集包含了1000个提示词工程转换示例，展示了如何将基本、低效的提示词转化为强大、高质量的提示词。它涵盖了多种任务（如信息提取、问答、创意写作、代码生成）和多种提示词技术（如角色扮演、思维链、上下文提示、少量示例）。  
  * **Prompt Engineering and Responses Dataset** 74: 该数据集包含5010条记录，用于探索不同类型的提示词（问题、命令、开放式陈述）如何影响生成的文本响应。它对于研究提示词有效性、对话代理、文本生成模型和情感分析等领域具有价值。  
  * **Kaggle Competitions Guide** 75: 尽管并非直接针对提示词工程，但Kaggle的机器学习竞赛指南提供了Python编程和数据分析的入门资源，这些技能对于理解和优化提示词至关重要。参与竞赛可以锻炼解决实际问题的能力。  
* Coursera Guided Projects:  
  Coursera的引导项目提供了短期的、实践性强的学习体验，适合快速上手。  
  * **Prompt Engineering with GPT: Programming for Custom Content** 76: 专门为程序员设计，通过Python编程和API使用GPT模型进行提示词工程实践，学习如何为自定义内容进行编程提示。  
  * **Customer Service with Python: Build a Chatbot using ChatGPT** 76: 学习如何结合Python和ChatGPT构建一个客服聊天机器人，涉及提示词工程、API使用和调试等实践技能。  
  * **Promptly for Beginners: Build a Generative AI App** 76: 引导初学者构建一个生成式AI应用，涵盖提示词工程、应用开发和部署的实际步骤。  
  * **Open AI for Beginners: Programmatic Prompting** 76: 教授如何通过编程方式使用OpenAI API进行提示词工程，适合希望自动化提示词交互的开发者。  
* Mini-Projects (Python):  
  对于编程背景的学习者，可以尝试一些小型Python项目来练习提示词工程。  
  * **构建AI聊天机器人 (Streamlit)** 77: 使用Streamlit构建一个简单的AI聊天机器人用户界面，类似OpenAI Playground，这有助于理解用户界面与LLM交互的逻辑。  
  * **文本加密生成器**: 编写一个能够对文本进行加密和解密的程序，这可以锻炼文本处理和算法设计能力，并思考如何通过提示词引导LLM进行类似操作 77。  
  * **互动问答应用**: 构建一个根据用户回答提供个性化结果的测验应用 77。这可以帮助理解如何设计多轮对话和基于用户输入的条件逻辑。

**结论与建议**

提示词工程已成为释放生成式AI潜力的关键，它通过精细化指令，将AI从模糊的通用工具转变为能够执行特定、高效任务的强大助手。其价值不仅体现在提升模型性能和用户体验，更在于实现AI应用在企业层面的可控性、灵活性和成本效益。

鉴于提示词工程的快速发展和复杂性，系统性学习路径至关重要。从基础概念到高级技术，再到特定领域应用和持续优化，每一步都构建在之前的基础上，形成一个连贯的知识体系。

对于希望系统性掌握“AI提示词工程”的初学者，尤其是有编程经验的学习者，建议采取以下策略：

1. **理论与实践结合**: 仅仅理解原理是不够的，必须通过大量的动手实践来巩固知识。积极参与GitHub上的开源项目、Kaggle挑战和Coursera的引导项目，将理论知识应用于实际场景。编程背景将是巨大的优势，应充分利用它来自动化提示词的测试和优化过程。  
2. **持续迭代与实验**: 提示词工程是一个实验性学科。拥抱试错，不断调整和优化提示词，是提升技能的关键。记录不同提示词版本的效果，并分析其优劣，从而逐步找到最佳实践。  
3. **关注最新发展**: AI领域发展迅速，新的模型、技术和最佳实践层出不穷。定期查阅最新论文、官方文档和在线课程，保持知识更新。特别是自动化提示词优化、检索增强生成和多模态AI代理等前沿方向，它们代表了提示词工程未来的发展趋势。  
4. **构建个人提示词库**: 随着学习的深入和实践经验的积累，开始组织和构建一套高效的、可复用的提示词模板。这些模板可以针对不同任务类型、模型和风格进行分类，这将极大地提高未来工作的效率和一致性。  
5. **重视评估与安全性**: 学习如何客观地评估提示词的有效性至关重要，包括使用人工评估和自动化指标。同时，始终关注AI的潜在偏见、幻觉和安全问题，确保负责任地使用AI，并设计能够减轻这些风险的提示词。

展望未来，提示词工程将进一步向自动化、代理化和多模态集成方向发展。未来的提示词工程师将不仅仅是“提示词的作者”，更是AI系统和工作流的“设计师”和“编排者”，需要结合软件工程和AI行为的深刻理解，以构建更智能、更强大、更安全的AI应用。

#### **Works cited**

1. What is Prompt Engineering? \- AI Prompt Engineering Explained \- AWS, accessed August 18, 2025, [https://aws.amazon.com/what-is/prompt-engineering/](https://aws.amazon.com/what-is/prompt-engineering/)  
2. What is prompt engineering? | SAP, accessed August 18, 2025, [https://www.sap.com/resources/what-is-prompt-engineering](https://www.sap.com/resources/what-is-prompt-engineering)  
3. The Evolution of Prompt Engineering | by Mattafrank \- Medium, accessed August 18, 2025, [https://medium.com/@Matthew\_Frank/the-evolution-of-prompt-engineering-7bda6c07f612](https://medium.com/@Matthew_Frank/the-evolution-of-prompt-engineering-7bda6c07f612)  
4. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models \- OpenReview, accessed August 18, 2025, [https://openreview.net/pdf?id=\_VjQlMeSB\_J](https://openreview.net/pdf?id=_VjQlMeSB_J)  
5. How Prompt Engineering Became the Brain of Agentic AI Systems \- Inclusion Cloud, accessed August 18, 2025, [https://inclusioncloud.com/insights/blog/the-evolution-of-prompt-engineering/](https://inclusioncloud.com/insights/blog/the-evolution-of-prompt-engineering/)  
6. 11 Prompt Engineering Best Practices Every Modern Dev Needs \- Mirascope, accessed August 18, 2025, [https://mirascope.com/blog/prompt-engineering-best-practices](https://mirascope.com/blog/prompt-engineering-best-practices)  
7. Creating Effective Prompts: Best Practices and Prompt Engineering, accessed August 18, 2025, [https://www.visiblethread.com/blog/creating-effective-prompts-best-practices-prompt-engineering-and-how-to-get-the-most-out-of-your-llm/](https://www.visiblethread.com/blog/creating-effective-prompts-best-practices-prompt-engineering-and-how-to-get-the-most-out-of-your-llm/)  
8. A Systematic Survey of Automatic Prompt Optimization Techniques, accessed August 18, 2025, [https://arxiv.org/abs/2502.16923](https://arxiv.org/abs/2502.16923)  
9. A Survey of Automatic Prompt Engineering: An Optimization Perspective \- arXiv, accessed August 18, 2025, [https://arxiv.org/html/2502.11560v1](https://arxiv.org/html/2502.11560v1)  
10. learn.microsoft.com, accessed August 18, 2025, [https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens\#:\~:text=Tokens%20are%20words%2C%20character%20sets,the%20first%20step%20in%20training.](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens#:~:text=Tokens%20are%20words%2C%20character%20sets,the%20first%20step%20in%20training.)  
11. Understanding tokens \- .NET | Microsoft Learn, accessed August 18, 2025, [https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens)  
12. www.ibm.com, accessed August 18, 2025, [https://www.ibm.com/think/topics/context-window\#:\~:text=The%20context%20window%20(or%20%E2%80%9Ccontext,of%20information%20into%20each%20output.](https://www.ibm.com/think/topics/context-window#:~:text=The%20context%20window%20\(or%20%E2%80%9Ccontext,of%20information%20into%20each%20output.)  
13. What is a context window? \- IBM, accessed August 18, 2025, [https://www.ibm.com/think/topics/context-window](https://www.ibm.com/think/topics/context-window)  
14. Understanding Temperature, Top P, and Maximum Length in LLMs, accessed August 18, 2025, [https://learnprompting.org/docs/intermediate/configuration\_hyperparameters](https://learnprompting.org/docs/intermediate/configuration_hyperparameters)  
15. LLM Settings \- Prompt Engineering Guide, accessed August 18, 2025, [https://www.promptingguide.ai/introduction/settings](https://www.promptingguide.ai/introduction/settings)  
16. Prompt design strategies | Gemini API | Google AI for Developers, accessed August 18, 2025, [https://ai.google.dev/gemini-api/docs/prompting-strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)  
17. Prompt Engineering for AI Guide | Google Cloud, accessed August 18, 2025, [https://cloud.google.com/discover/what-is-prompt-engineering](https://cloud.google.com/discover/what-is-prompt-engineering)  
18. 26 Prompt Engineering Principles for 2024 | by Dan Cleary \- Medium, accessed August 18, 2025, [https://medium.com/@dan\_43009/26-prompt-engineering-principles-for-2024-775099ddfe94](https://medium.com/@dan_43009/26-prompt-engineering-principles-for-2024-775099ddfe94)  
19. Prompt Engineering Principles for 2024 \- PromptHub, accessed August 18, 2025, [https://www.prompthub.us/blog/prompt-engineering-principles-for-2024](https://www.prompthub.us/blog/prompt-engineering-principles-for-2024)  
20. Prompt engineering \- OpenAI API \- OpenAI platform, accessed August 18, 2025, [https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions](https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions)  
21. A Comprehensive Review of Retrieval-Augmented Generation (RAG): Key Challenges and Future Directions \- arXiv, accessed August 18, 2025, [https://arxiv.org/pdf/2410.12837](https://arxiv.org/pdf/2410.12837)  
22. Best Prompts for Text Summarization: Guide to AI Summaries \- PromptLayer, accessed August 18, 2025, [https://blog.promptlayer.com/best-prompts-for-text-summarization-guide-to-ai-summaries/](https://blog.promptlayer.com/best-prompts-for-text-summarization-guide-to-ai-summaries/)  
23. Effective Prompts for AI: The Essentials \- MIT Sloan Teaching & Learning Technologies, accessed August 18, 2025, [https://mitsloanedtech.mit.edu/ai/basics/effective-prompts/](https://mitsloanedtech.mit.edu/ai/basics/effective-prompts/)  
24. Prompt engineering \- Hugging Face, accessed August 18, 2025, [https://huggingface.co/docs/transformers/main/tasks/prompting](https://huggingface.co/docs/transformers/main/tasks/prompting)  
25. Prompt Engineering Techniques | IBM, accessed August 18, 2025, [https://www.ibm.com/think/topics/prompt-engineering-techniques](https://www.ibm.com/think/topics/prompt-engineering-techniques)  
26. Self-Harmonized Chain of Thought \- arXiv, accessed August 18, 2025, [https://arxiv.org/html/2409.04057v2](https://arxiv.org/html/2409.04057v2)  
27. Zero-Shot Verification-guided Chain of Thoughts \- arXiv, accessed August 18, 2025, [https://arxiv.org/html/2501.13122v1](https://arxiv.org/html/2501.13122v1)  
28. Chain of Thought Prompting Explained (with examples) \- Codecademy, accessed August 18, 2025, [https://www.codecademy.com/article/chain-of-thought-cot-prompting](https://www.codecademy.com/article/chain-of-thought-cot-prompting)  
29. Chain-of-Thought Prompting: A Comprehensive Analysis of Reasoning Techniques in Large Language Models | by Pier-Jean Malandrino | Scub-Lab, accessed August 18, 2025, [https://lab.scub.net/chain-of-thought-prompting-a-comprehensive-analysis-of-reasoning-techniques-in-large-language-b67fdd2eb72a](https://lab.scub.net/chain-of-thought-prompting-a-comprehensive-analysis-of-reasoning-techniques-in-large-language-b67fdd2eb72a)  
30. Chain-of-Thought Prompting | Prompt Engineering Guide, accessed August 18, 2025, [https://www.promptingguide.ai/techniques/cot](https://www.promptingguide.ai/techniques/cot)  
31. Strategies For Effective Prompt Engineering \- Neptune.ai, accessed August 18, 2025, [https://neptune.ai/blog/prompt-engineering-strategies](https://neptune.ai/blog/prompt-engineering-strategies)  
32. \[2401.14295\] Demystifying Chains, Trees, and Graphs of Thoughts \- arXiv, accessed August 18, 2025, [https://arxiv.org/abs/2401.14295](https://arxiv.org/abs/2401.14295)  
33. ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving \- arXiv, accessed August 18, 2025, [https://arxiv.org/html/2505.12717v1](https://arxiv.org/html/2505.12717v1)  
34. What is Tree Of Thoughts Prompting? \- IBM, accessed August 18, 2025, [https://www.ibm.com/think/topics/tree-of-thoughts](https://www.ibm.com/think/topics/tree-of-thoughts)  
35. Beginner's Guide To Tree Of Thoughts Prompting (With Examples) | Zero To Mastery, accessed August 18, 2025, [https://zerotomastery.io/blog/tree-of-thought-prompting/](https://zerotomastery.io/blog/tree-of-thought-prompting/)  
36. Tree of Thoughts: Deliberate Problem Solving with Large Language Models \- OpenReview, accessed August 18, 2025, [https://openreview.net/forum?id=5Xc1ecxO1h](https://openreview.net/forum?id=5Xc1ecxO1h)  
37. Tree of Thoughts (ToT) \- Prompt Engineering Guide, accessed August 18, 2025, [https://www.promptingguide.ai/techniques/tot](https://www.promptingguide.ai/techniques/tot)  
38. Retrieval-Augmented Generation for Large Language ... \- arXiv, accessed August 18, 2025, [https://arxiv.org/pdf/2312.10997](https://arxiv.org/pdf/2312.10997)  
39. What is Retrieval-Augmented Generation (RAG)? \- Google Cloud, accessed August 18, 2025, [https://cloud.google.com/use-cases/retrieval-augmented-generation](https://cloud.google.com/use-cases/retrieval-augmented-generation)  
40. 8 Retrieval Augmented Generation (RAG) Architectures You Should Know in 2025, accessed August 18, 2025, [https://humanloop.com/blog/rag-architectures](https://humanloop.com/blog/rag-architectures)  
41. RETRIEVAL-AUGMENTED GENERATION: ARCHITECTURE, TECHNIQUES, AND EVALUATIONS \- Jomard Publishing, accessed August 18, 2025, [https://api.jomardpublishing.com/api/main/articles/view?source=storage/journals/journal-of-modern-technology-and-engineering/issues/pdf/2025/retrieval-augmented-generation-architecture-techniques-and-evaluations.pdf](https://api.jomardpublishing.com/api/main/articles/view?source=storage/journals/journal-of-modern-technology-and-engineering/issues/pdf/2025/retrieval-augmented-generation-architecture-techniques-and-evaluations.pdf)  
42. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks \- ResearchGate, accessed August 18, 2025, [https://www.researchgate.net/publication/355023301\_Retrieval-Augmented\_Generation\_for\_Knowledge-Intensive\_NLP\_Tasks](https://www.researchgate.net/publication/355023301_Retrieval-Augmented_Generation_for_Knowledge-Intensive_NLP_Tasks)  
43. What is Prompt Optimization? | IBM, accessed August 18, 2025, [https://www.ibm.com/think/topics/prompt-optimization](https://www.ibm.com/think/topics/prompt-optimization)  
44. Exploring Prompt Optimization \- LangChain Blog, accessed August 18, 2025, [https://blog.langchain.com/exploring-prompt-optimization/](https://blog.langchain.com/exploring-prompt-optimization/)  
45. Prompt Optimization Using Few-Shot Prompting: Proven Tactics \- Arize AI, accessed August 18, 2025, [https://arize.com/blog/prompt-optimization-few-shot-prompting/](https://arize.com/blog/prompt-optimization-few-shot-prompting/)  
46. \[2501.01237\] Self-Refinement Strategies for LLM-based Product Attribute Value Extraction, accessed August 18, 2025, [https://arxiv.org/abs/2501.01237](https://arxiv.org/abs/2501.01237)  
47. Prompt Chaining or Stepwise Prompt? Refinement in Text ..., accessed August 18, 2025, [https://arxiv.org/pdf/2406.00507](https://arxiv.org/pdf/2406.00507)  
48. Iterative Refinement with Self-Feedback \- OpenReview, accessed August 18, 2025, [https://openreview.net/pdf?id=S37hOerQLB](https://openreview.net/pdf?id=S37hOerQLB)  
49. How to Build a Self-Refining LLM Agent — Step-by-Step with Examples \- Medium, accessed August 18, 2025, [https://medium.com/@praveencs87/how-to-build-a-self-refining-llm-agent-step-by-step-with-examples-e309d9df0eae](https://medium.com/@praveencs87/how-to-build-a-self-refining-llm-agent-step-by-step-with-examples-e309d9df0eae)  
50. Prompt engineering for GitHub Copilot Chat \- GitHub Docs, accessed August 18, 2025, [https://docs.github.com/en/copilot/concepts/prompt-engineering](https://docs.github.com/en/copilot/concepts/prompt-engineering)  
51. Prompt Engineering for Code Generation: Examples & Best Practices, accessed August 18, 2025, [https://margabagus.com/prompt-engineering-code-generation-practices/](https://margabagus.com/prompt-engineering-code-generation-practices/)  
52. Learn AI Prompting with Google Prompting Essentials, accessed August 18, 2025, [https://grow.google/prompting-essentials/](https://grow.google/prompting-essentials/)  
53. Prompt Engineering in Image Generation vs Textual Generation : r/PromptEngineering \- Reddit, accessed August 18, 2025, [https://www.reddit.com/r/PromptEngineering/comments/1akdhdp/prompt\_engineering\_in\_image\_generation\_vs\_textual/](https://www.reddit.com/r/PromptEngineering/comments/1akdhdp/prompt_engineering_in_image_generation_vs_textual/)  
54. General Tips for Designing Prompts | Prompt Engineering Guide, accessed August 18, 2025, [https://www.promptingguide.ai/introduction/tips](https://www.promptingguide.ai/introduction/tips)  
55. 10 Prompt Engineering Best Practices | by Pieces \- Medium, accessed August 18, 2025, [https://pieces.medium.com/10-prompt-engineering-best-practices-a166fe2f101b](https://pieces.medium.com/10-prompt-engineering-best-practices-a166fe2f101b)  
56. Image prompt engineering techniques \- Azure OpenAI | Microsoft Learn, accessed August 18, 2025, [https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/gpt-4-v-prompt-engineering](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/gpt-4-v-prompt-engineering)  
57. LLM Evaluation: Frameworks, Metrics, and Best Practices ..., accessed August 18, 2025, [https://www.superannotate.com/blog/llm-evaluation-guide](https://www.superannotate.com/blog/llm-evaluation-guide)  
58. LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide \- Confident AI, accessed August 18, 2025, [https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)  
59. A/B Testing for Prompt Optimization: A Comprehensive Guide | by ..., accessed August 18, 2025, [https://medium.com/@support\_11335/a-b-testing-for-prompt-optimization-a-comprehensive-guide-93fb33d29c78](https://medium.com/@support_11335/a-b-testing-for-prompt-optimization-a-comprehensive-guide-93fb33d29c78)  
60. The Ultimate Guide to Prompt Engineering \- Every, accessed August 18, 2025, [https://every.to/p/the-ultimate-guide-to-prompt-engineering?p=cdd92917ff31dd66f1a51196fa15c4f1391b72c360c634cc82f654cb5434d749](https://every.to/p/the-ultimate-guide-to-prompt-engineering?p=cdd92917ff31dd66f1a51196fa15c4f1391b72c360c634cc82f654cb5434d749)  
61. Prompt Versioning: Best Practices \- Ghost, accessed August 18, 2025, [https://latitude-blog.ghost.io/blog/prompt-versioning-best-practices/](https://latitude-blog.ghost.io/blog/prompt-versioning-best-practices/)  
62. Prompt Versioning & Management Guide for Building AI Features \- LaunchDarkly, accessed August 18, 2025, [https://launchdarkly.com/blog/prompt-versioning-and-management/](https://launchdarkly.com/blog/prompt-versioning-and-management/)  
63. 8 Best Prompt Engineering Tools in 2025 \- Mirascope, accessed August 18, 2025, [https://mirascope.com/blog/prompt-engineering-tools](https://mirascope.com/blog/prompt-engineering-tools)  
64. GPT-4.1 Prompting Guide \- OpenAI Cookbook, accessed August 18, 2025, [https://cookbook.openai.com/examples/gpt4-1\_prompting\_guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)  
65. Best Prompt Engineering Courses & Certificates Online \[2025\] \- Coursera, accessed August 18, 2025, [https://www.coursera.org/courses?query=prompt%20engineering](https://www.coursera.org/courses?query=prompt+engineering)  
66. Generative AI: Prompt Engineering Basics by IBM \- Coursera, accessed August 18, 2025, [https://www.coursera.org/learn/generative-ai-prompt-engineering-for-everyone](https://www.coursera.org/learn/generative-ai-prompt-engineering-for-everyone)  
67. Top Prompt Engineering Courses Online \- Updated \[August 2025\] \- Udemy, accessed August 18, 2025, [https://www.udemy.com/topic/prompt-engineering/](https://www.udemy.com/topic/prompt-engineering/)  
68. 10 Best Online Prompt Engineering Courses \[Free & Paid\] with Certificates, accessed August 18, 2025, [https://learnprompting.org/blog/prompt\_engineering\_courses](https://learnprompting.org/blog/prompt_engineering_courses)  
69. Prompt Engineering: A Practical Example \- Real Python, accessed August 18, 2025, [https://realpython.com/practical-prompt-engineering/](https://realpython.com/practical-prompt-engineering/)  
70. NirDiamant/Prompt\_Engineering: This repository offers a comprehensive collection of tutorials and implementations for Prompt Engineering techniques, ranging from fundamental concepts to advanced strategies. It serves as an essential resource for mastering the art of effectively communicating with and leveraging large language models in AI applications. \- GitHub, accessed August 18, 2025, [https://github.com/NirDiamant/Prompt\_Engineering](https://github.com/NirDiamant/Prompt_Engineering)  
71. Introducing the Prompt Engineering Repository: Nearly 4,000 Stars on GitHub Link to Repo : r/LangChain \- Reddit, accessed August 18, 2025, [https://www.reddit.com/r/LangChain/comments/1juf57p/introducing\_the\_prompt\_engineering\_repository/](https://www.reddit.com/r/LangChain/comments/1juf57p/introducing_the_prompt_engineering_repository/)  
72. prompt-engineering · GitHub Topics, accessed August 18, 2025, [https://github.com/topics/prompt-engineering](https://github.com/topics/prompt-engineering)  
73. Prompt Engineering Dataset \- Kaggle, accessed August 18, 2025, [https://www.kaggle.com/datasets/austinfairbanks/prompt-engineering-dataset](https://www.kaggle.com/datasets/austinfairbanks/prompt-engineering-dataset)  
74. Prompt Engineering and Responses Dataset \- Kaggle, accessed August 18, 2025, [https://www.kaggle.com/datasets/antrixsh/prompt-engineering-and-responses-dataset](https://www.kaggle.com/datasets/antrixsh/prompt-engineering-and-responses-dataset)  
75. Kaggle Competitions Guide, accessed August 18, 2025, [https://www.kaggle.com/learn-guide/kaggle-competitions](https://www.kaggle.com/learn-guide/kaggle-competitions)  
76. Prompt Engineering Practice Projects & Exercises \[2025\] \- Coursera, accessed August 18, 2025, [https://www.coursera.org/courses?query=prompt%20engineering\&productTypeDescription=Guided%20Projects](https://www.coursera.org/courses?query=prompt+engineering&productTypeDescription=Guided+Projects)  
77. Python Projects for Beginners: 60+ Ideas to Build Your Portfolio \- Dataquest, accessed August 18, 2025, [https://www.dataquest.io/blog/python-projects-for-beginners/](https://www.dataquest.io/blog/python-projects-for-beginners/)