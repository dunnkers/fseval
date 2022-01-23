"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[2088],{8215:function(e,t,n){var a=n(7294);t.Z=function(e){var t=e.children,n=e.hidden,r=e.className;return a.createElement("div",{role:"tabpanel",hidden:n,className:r},t)}},6396:function(e,t,n){n.d(t,{Z:function(){return f}});var a=n(7462),r=n(7294),o=n(2389),l=n(9443);var i=function(){var e=(0,r.useContext)(l.Z);if(null==e)throw new Error('"useUserPreferencesContext" is used outside of "Layout" component.');return e},u=n(3616),s=n(6010),c="tabItem_vU9c";function d(e){var t,n,o,l=e.lazy,d=e.block,f=e.defaultValue,g=e.values,p=e.groupId,v=e.className,m=r.Children.map(e.children,(function(e){if((0,r.isValidElement)(e)&&void 0!==e.props.value)return e;throw new Error("Docusaurus error: Bad <Tabs> child <"+("string"==typeof e.type?e.type:e.type.name)+'>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.')})),b=null!=g?g:m.map((function(e){var t=e.props;return{value:t.value,label:t.label,attributes:t.attributes}})),h=(0,u.lx)(b,(function(e,t){return e.value===t.value}));if(h.length>0)throw new Error('Docusaurus error: Duplicate values "'+h.map((function(e){return e.value})).join(", ")+'" found in <Tabs>. Every value needs to be unique.');var k=null===f?f:null!=(t=null!=f?f:null==(n=m.find((function(e){return e.props.default})))?void 0:n.props.value)?t:null==(o=m[0])?void 0:o.props.value;if(null!==k&&!b.some((function(e){return e.value===k})))throw new Error('Docusaurus error: The <Tabs> has a defaultValue "'+k+'" but none of its children has the corresponding value. Available values are: '+b.map((function(e){return e.value})).join(", ")+". If you intend to show no default tab, use defaultValue={null} instead.");var N=i(),y=N.tabGroupChoices,C=N.setTabGroupChoices,w=(0,r.useState)(k),E=w[0],S=w[1],T=[],x=(0,u.o5)().blockElementScrollPositionUntilNextRender;if(null!=p){var _=y[p];null!=_&&_!==E&&b.some((function(e){return e.value===_}))&&S(_)}var Z=function(e){var t=e.currentTarget,n=T.indexOf(t),a=b[n].value;a!==E&&(x(t),S(a),null!=p&&C(p,a))},A=function(e){var t,n=null;switch(e.key){case"ArrowRight":var a=T.indexOf(e.currentTarget)+1;n=T[a]||T[0];break;case"ArrowLeft":var r=T.indexOf(e.currentTarget)-1;n=T[r]||T[T.length-1]}null==(t=n)||t.focus()};return r.createElement("div",{className:"tabs-container"},r.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,s.Z)("tabs",{"tabs--block":d},v)},b.map((function(e){var t=e.value,n=e.label,o=e.attributes;return r.createElement("li",(0,a.Z)({role:"tab",tabIndex:E===t?0:-1,"aria-selected":E===t,key:t,ref:function(e){return T.push(e)},onKeyDown:A,onFocus:Z,onClick:Z},o,{className:(0,s.Z)("tabs__item",c,null==o?void 0:o.className,{"tabs__item--active":E===t})}),null!=n?n:t)}))),l?(0,r.cloneElement)(m.filter((function(e){return e.props.value===E}))[0],{className:"margin-vert--md"}):r.createElement("div",{className:"margin-vert--md"},m.map((function(e,t){return(0,r.cloneElement)(e,{key:t,hidden:e.props.value!==E})}))))}function f(e){var t=(0,o.Z)();return r.createElement(d,(0,a.Z)({key:String(t)},e))}},5404:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return i},contentTitle:function(){return u},metadata:function(){return s},toc:function(){return c},default:function(){return f}});var a=n(7462),r=n(3366),o=(n(7294),n(3905)),l=(n(6396),n(8215),n(9055),["components"]),i={sidebar_position:7,title:"fseval.config.StorageConfig"},u="StorageConfig",s={unversionedId:"config/StorageConfig",id:"config/StorageConfig",title:"fseval.config.StorageConfig",description:"Allows you to define a storage for loading and saving cached estimators, among other",source:"@site/docs/config/StorageConfig.mdx",sourceDirName:"config",slug:"/config/StorageConfig",permalink:"/fseval/docs/config/StorageConfig",editUrl:"https://github.com/dunnkers/fseval/tree/website/docs/config/StorageConfig.mdx",tags:[],version:"current",sidebarPosition:7,frontMatter:{sidebar_position:7,title:"fseval.config.StorageConfig"},sidebar:"tutorialSidebar",previous:{title:"fseval.config.EstimatorConfig",permalink:"/fseval/docs/config/EstimatorConfig"},next:{title:"LocalStorageConfig",permalink:"/fseval/docs/config/storage/local"}},c=[{value:"Available storages",id:"available-storages",children:[],level:2}],d={toc:c};function f(e){var t=e.components,n=(0,r.Z)(e,l);return(0,o.kt)("wrapper",(0,a.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"storageconfig"},"StorageConfig"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"class fseval.config.StorageConfig(\n    load_dir: Optional[str]=None,\n    save_dir: Optional[str]=None,\n)\n")),(0,o.kt)("p",null,"Allows you to define a storage for loading and saving cached estimators, among other\nfiles, like the hydra and fseval configuration in YAML."),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Attributes"),":"),(0,o.kt)("table",null,(0,o.kt)("thead",{parentName:"table"},(0,o.kt)("tr",{parentName:"thead"},(0,o.kt)("th",{parentName:"tr",align:null}),(0,o.kt)("th",{parentName:"tr",align:null}))),(0,o.kt)("tbody",{parentName:"table"},(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("inlineCode",{parentName:"td"},"load_dir")," : Optional","[str]"),(0,o.kt)("td",{parentName:"tr",align:null},"Defines a path to load files from. Must point to exactly the directory containing the files, i.e. you should not point to a higher-level directory than where the files are. Path can be relative or absolute, but an absolute path is recommended.")),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null},(0,o.kt)("inlineCode",{parentName:"td"},"save_dir")," : Optional","[str]"),(0,o.kt)("td",{parentName:"tr",align:null},"The directory to save files to. Can be relative or absolute.")),(0,o.kt)("tr",{parentName:"tbody"},(0,o.kt)("td",{parentName:"tr",align:null}),(0,o.kt)("td",{parentName:"tr",align:null})))),(0,o.kt)("h2",{id:"available-storages"},"Available storages"))}f.isMDXComponent=!0}}]);